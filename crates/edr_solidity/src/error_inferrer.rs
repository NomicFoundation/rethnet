use std::{borrow::Cow, collections::HashSet, mem};

use alloy_dyn_abi::{DynSolValue, JsonAbiExt};
use edr_eth::{bytecode::opcode::OpCode, hex, spec::HaltReasonTrait, U256};
use semver::{Version, VersionReq};

use crate::{
    build_model::{
        ContractFunction, ContractFunctionType, ContractKind, ContractMetadata,
        ContractMetadataError, Instruction, JumpType, SourceLocation,
    },
    nested_trace::{
        CallMessage, CreateMessage, CreateOrCallMessageRef, NestedTrace, NestedTraceStep,
    },
    return_data::ReturnData,
    solidity_stack_trace::{
        SourceReference, StackTraceEntry, CONSTRUCTOR_FUNCTION_NAME, FALLBACK_FUNCTION_NAME,
        RECEIVE_FUNCTION_NAME,
    },
};

const FIRST_SOLC_VERSION_CREATE_PARAMS_VALIDATION: Version = Version::new(0, 5, 9);
const FIRST_SOLC_VERSION_RECEIVE_FUNCTION: Version = Version::new(0, 6, 0);
const FIRST_SOLC_VERSION_WITH_UNMAPPED_REVERTS: &str = "0.6.3";

/// Specifies whether a heuristic was applied and modified the stack trace.
///
/// Think of it as happy [`Result`] - the [`Heuristic::Hit`] should be
/// propagated to the caller.
#[must_use]
pub enum Heuristic {
    /// The heuristic was applied and modified the stack trace.
    Hit(Vec<StackTraceEntry>),
    /// The heuristic did not apply; the stack trace is unchanged.
    Miss(Vec<StackTraceEntry>),
}

/// Data that is used to infer the stack trace of a submessage.
#[derive(Clone, Debug)]
pub(crate) struct SubmessageData<HaltReasonT: HaltReasonTrait> {
    pub message_trace: NestedTrace<HaltReasonT>,
    pub stacktrace: Vec<StackTraceEntry>,
    pub step_index: u32,
}

/// Errors that can occur during the inference of the stack trace.
#[derive(Debug, thiserror::Error)]
pub enum InferrerError<HaltReasonT: HaltReasonTrait> {
    /// Errors that can occur when decoding the ABI.
    #[error("{0}")]
    Abi(String),
    /// Errors that can occur when decoding the contract metadata.
    #[error(transparent)]
    ContractMetadata(#[from] ContractMetadataError),
    /// Invalid input or logic error: Expected an EVM step.
    #[error("Expected EVM step")]
    ExpectedEvmStep,
    /// Invalid input or logic error: Missing contract metadata.
    #[error("Missing contract")]
    MissingContract,
    /// Invalid input or logic error: The call trace has no functionJumpdest but
    /// has already jumped into a function.
    #[error("call trace has no functionJumpdest but has already jumped into a function")]
    MissingFunctionJumpDest(CallMessage<HaltReasonT>),
    /// Invalid input or logic error: Missing source reference.
    #[error("Missing source reference")]
    MissingSourceReference,
    /// Serde JSON error.
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    /// Solidity types error.
    #[error(transparent)]
    SolidityTypes(#[from] alloy_sol_types::Error),
}

// Automatic conversion from `alloy_dyn_abi::Error` to `InferrerError` is not
// possible due to unsatisifed trait bounds.
impl<HaltReasonT: HaltReasonTrait> From<alloy_dyn_abi::Error> for InferrerError<HaltReasonT> {
    fn from(err: alloy_dyn_abi::Error) -> Self {
        Self::Abi(err.to_string())
    }
}

pub(crate) fn filter_redundant_frames<HaltReasonT: HaltReasonTrait>(
    stacktrace: Vec<StackTraceEntry>,
) -> Result<Vec<StackTraceEntry>, InferrerError<HaltReasonT>> {
    // To work around the borrow checker, we'll collect the indices of the frames we
    // want to keep. We can't clone the frames, because some of them contain
    // non-Clone `ClassInstance`s`
    let retained_indices: HashSet<_> = stacktrace
        .iter()
        .enumerate()
        .filter(|(idx, frame)| {
            let next_frame = stacktrace.get(idx + 1);
            let next_next_frame = stacktrace.get(idx + 2);

            let Some(next_frame) = next_frame else {
                return true;
            };

            // we can only filter frames if we know their sourceReference
            // and the one from the next frame
            let (Some(frame_source), Some(next_frame_source)) =
                (frame.source_reference(), next_frame.source_reference())
            else {
                return true;
            };

            // look TWO frames ahead to determine if this is a specific occurrence of
            // a redundant CALLSTACK_ENTRY frame observed when using Solidity 0.8.5:
            match (&frame, next_next_frame) {
                (
                    StackTraceEntry::CallstackEntry {
                        source_reference, ..
                    },
                    Some(StackTraceEntry::ReturndataSizeError {
                        source_reference: next_next_source_reference,
                        ..
                    }),
                ) if source_reference.range == next_next_source_reference.range
                    && source_reference.line == next_next_source_reference.line =>
                {
                    return false;
                }
                _ => {}
            }

            if frame_source.function.as_deref() == Some("constructor")
                && next_frame_source.function.as_deref() != Some("constructor")
            {
                return true;
            }

            // this is probably a recursive call
            if *idx > 0
                && mem::discriminant(*frame) == mem::discriminant(next_frame)
                && frame_source.range == next_frame_source.range
                && frame_source.line == next_frame_source.line
            {
                return true;
            }

            if frame_source.range.0 <= next_frame_source.range.0
                && frame_source.range.1 >= next_frame_source.range.1
            {
                return false;
            }

            true
        })
        .map(|(idx, _)| idx)
        .collect();

    Ok(stacktrace
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| retained_indices.contains(idx))
        .map(|(_, frame)| frame)
        .collect())
}

pub(crate) fn infer_after_tracing<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
    function_jumpdests: &[&Instruction],
    jumped_into_function: bool,
    last_submessage_data: Option<SubmessageData<HaltReasonT>>,
) -> Result<Vec<StackTraceEntry>, InferrerError<HaltReasonT>> {
    /// Convenience macro to early return the result if a heuristic hits.
    macro_rules! return_if_hit {
        ($heuristic: expr) => {
            match $heuristic {
                Heuristic::Hit(stacktrace) => return Ok(stacktrace),
                Heuristic::Miss(stacktrace) => stacktrace,
            }
        };
    }

    let result = check_last_submessage(trace, stacktrace, last_submessage_data)?;
    let stacktrace = return_if_hit!(result);

    let result = check_failed_last_call(trace, stacktrace)?;
    let stacktrace = return_if_hit!(result);

    let result =
        check_last_instruction(trace, stacktrace, function_jumpdests, jumped_into_function)?;
    let stacktrace = return_if_hit!(result);

    let result = check_non_contract_called(trace, stacktrace)?;
    let stacktrace = return_if_hit!(result);

    let result = check_solidity_0_6_3_unmapped_revert(trace, stacktrace)?;
    let stacktrace = return_if_hit!(result);

    if let Some(result) = check_contract_too_large(trace)? {
        return Ok(result);
    }

    let stacktrace = other_execution_error_stacktrace(trace, stacktrace)?;
    Ok(stacktrace)
}

pub(crate) fn infer_before_tracing_call_message<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<Option<Vec<StackTraceEntry>>, InferrerError<HaltReasonT>> {
    if is_direct_library_call(trace)? {
        return Ok(Some(get_direct_library_call_error_stack_trace(trace)?));
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let called_function =
        contract.get_function_from_selector(trace.calldata.get(..4).unwrap_or(&trace.calldata[..]));

    if let Some(called_function) = called_function {
        if is_function_not_payable_error(trace, called_function)? {
            return Ok(Some(vec![StackTraceEntry::FunctionNotPayableError {
                source_reference: get_function_start_source_reference(
                    trace.into(),
                    called_function,
                )?,
                value: trace.value,
            }]));
        }
    }

    let called_function = called_function.map(AsRef::as_ref);

    if is_missing_function_and_fallback_error(trace, called_function)? {
        let source_reference = get_contract_start_without_function_source_reference(trace.into())?;

        if empty_calldata_and_no_receive(trace)? {
            return Ok(Some(vec![StackTraceEntry::MissingFallbackOrReceiveError {
                source_reference,
            }]));
        }

        return Ok(Some(vec![
            StackTraceEntry::UnrecognizedFunctionWithoutFallbackError { source_reference },
        ]));
    }

    if is_fallback_not_payable_error(trace, called_function)? {
        let source_reference = get_fallback_start_source_reference(trace)?;

        if empty_calldata_and_no_receive(trace)? {
            return Ok(Some(vec![
                StackTraceEntry::FallbackNotPayableAndNoReceiveError {
                    source_reference,
                    value: trace.value,
                },
            ]));
        }

        return Ok(Some(vec![StackTraceEntry::FallbackNotPayableError {
            source_reference,
            value: trace.value,
        }]));
    }

    Ok(None)
}

pub(crate) fn infer_before_tracing_create_message<HaltReasonT: HaltReasonTrait>(
    trace: &CreateMessage<HaltReasonT>,
) -> Result<Option<Vec<StackTraceEntry>>, InferrerError<HaltReasonT>> {
    if is_constructor_not_payable_error(trace)? {
        return Ok(Some(vec![StackTraceEntry::FunctionNotPayableError {
            source_reference: get_constructor_start_source_reference(trace)?,
            value: trace.value,
        }]));
    }

    if is_constructor_invalid_arguments_error(trace)? {
        return Ok(Some(vec![StackTraceEntry::InvalidParamsError {
            source_reference: get_constructor_start_source_reference(trace)?,
        }]));
    }

    Ok(None)
}

pub(crate) fn instruction_to_callstack_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    contract_meta: &ContractMetadata,
    inst: &Instruction,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let contract = contract_meta.contract.read();

    // This means that a jump is made from within an internal solc function.
    // These are normally made from yul code, so they don't map to any Solidity
    // function
    let inst_location = match &inst.location {
        None => {
            let location = &contract.location;
            let file = location.file();
            let file = file.read();

            return Ok(StackTraceEntry::InternalFunctionCallstackEntry {
                pc: inst.pc,
                source_reference: SourceReference {
                    source_name: file.source_name.clone(),
                    source_content: file.content.clone(),
                    contract: Some(contract.name.clone()),
                    function: None,
                    line: location.get_starting_line_number(),
                    range: (location.offset, location.offset + location.length),
                },
            });
        }
        Some(inst_location) => inst_location,
    };

    if let Some(func) = inst_location.get_containing_function() {
        let source_reference =
            source_location_to_source_reference(contract_meta, Some(inst_location))
                .ok_or(InferrerError::MissingSourceReference)?;

        return Ok(StackTraceEntry::CallstackEntry {
            source_reference,
            function_type: func.r#type,
        });
    };

    let file = inst_location.file();
    let file = file.read();

    Ok(StackTraceEntry::CallstackEntry {
        source_reference: SourceReference {
            function: None,
            contract: Some(contract.name.clone()),
            source_name: file.source_name.clone(),
            source_content: file.content.clone(),
            line: inst_location.get_starting_line_number(),
            range: (
                inst_location.offset,
                inst_location.offset + inst_location.length,
            ),
        },
        function_type: ContractFunctionType::Function,
    })
}

fn call_instruction_to_call_failed_to_execute_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    contract_meta: &ContractMetadata,
    call_inst: &Instruction,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let location = call_inst.location.as_deref();

    let source_reference = source_location_to_source_reference(contract_meta, location)
        .ok_or(InferrerError::MissingSourceReference)?;

    // Calls only happen within functions
    Ok(StackTraceEntry::CallFailedError { source_reference })
}

fn check_contract_too_large<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Option<Vec<StackTraceEntry>>, InferrerError<HaltReasonT>> {
    if let CreateOrCallMessageRef::Create(create) = trace {
        if create.exit.is_contract_too_large_error() {
            return Ok(Some(vec![StackTraceEntry::ContractTooLargeError {
                source_reference: Some(get_constructor_start_source_reference(create)?),
            }]));
        }
    }
    Ok(None)
}

fn check_custom_errors<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
    last_instruction: &Instruction,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();
    let return_data = trace.return_data();

    let return_data = ReturnData::new(return_data.clone());

    if return_data.is_empty() || return_data.is_error_return_data() {
        // if there is no return data, or if it's a Error(string),
        // then it can't be a custom error
        return Ok(Heuristic::Miss(stacktrace));
    }

    let raw_return_data = hex::encode(&*return_data.value);
    let mut error_message =
        format!("reverted with an unrecognized custom error (return data: 0x{raw_return_data})",);

    for custom_error in &contract.custom_errors {
        if return_data.matches_selector(custom_error.selector) {
            // if the return data matches a custom error in the called contract,
            // we format the message using the returnData and the custom error instance
            let decoded = custom_error.decode_error_data(&return_data.value)?;

            let params = decoded
                .body
                .iter()
                .map(format_dyn_sol_value)
                .collect::<Vec<_>>();

            error_message = format!(
                "reverted with custom error '{name}({params})'",
                name = custom_error.name,
                params = params.join(", ")
            );

            break;
        }
    }

    let mut stacktrace = stacktrace;
    stacktrace.push(
        instruction_within_function_to_custom_error_stack_trace_entry(
            trace,
            last_instruction,
            error_message,
        )?,
    );

    fix_initial_modifier(trace, stacktrace).map(Heuristic::Hit)
}

/// Check if the last call/create that was done failed.
fn check_failed_last_call<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    if steps.is_empty() {
        return Ok(Heuristic::Miss(stacktrace));
    }

    for step_index in (0..steps.len() - 1).rev() {
        let (step, next_step) = match &steps[step_index..][..2] {
            &[NestedTraceStep::Evm(ref step), ref next_step] => (step, next_step),
            _ => return Ok(Heuristic::Miss(stacktrace)),
        };

        let inst = contract_meta.get_instruction(step.pc)?;

        if let (OpCode::CALL | OpCode::CREATE, NestedTraceStep::Evm(_)) = (inst.opcode, next_step) {
            if is_call_failed_error(trace, step_index as u32, inst)? {
                let mut inferred_stacktrace = stacktrace.clone();
                inferred_stacktrace.push(
                    call_instruction_to_call_failed_to_execute_stack_trace_entry(
                        &contract_meta,
                        inst,
                    )?,
                );

                return Ok(Heuristic::Hit(fix_initial_modifier(
                    trace,
                    inferred_stacktrace,
                )?));
            }
        }
    }

    Ok(Heuristic::Miss(stacktrace))
}

fn check_last_instruction<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
    function_jumpdests: &[&Instruction],
    jumped_into_function: bool,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    if steps.is_empty() {
        return Ok(Heuristic::Miss(stacktrace));
    }

    let last_step = match steps.last() {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => panic!("This should not happen: MessageTrace ends with a subtrace"),
    };

    let last_instruction = contract_meta.get_instruction(last_step.pc)?;

    let revert_or_invalid_stacktrace = check_revert_or_invalid_opcode(
        trace,
        stacktrace,
        last_instruction,
        function_jumpdests,
        jumped_into_function,
    )?;
    let stacktrace = match revert_or_invalid_stacktrace {
        hit @ Heuristic::Hit(..) => return Ok(hit),
        Heuristic::Miss(stacktrace) => stacktrace,
    };

    let (CreateOrCallMessageRef::Call(trace @ CallMessage { calldata, .. }), false) =
        (trace, jumped_into_function)
    else {
        return Ok(Heuristic::Miss(stacktrace));
    };

    if has_failed_inside_the_fallback_function(trace)?
        || has_failed_inside_the_receive_function(trace)?
    {
        let frame = instruction_within_function_to_revert_stack_trace_entry(
            CreateOrCallMessageRef::Call(trace),
            last_instruction,
        )?;

        return Ok(Heuristic::Hit(vec![frame]));
    }

    // Sometimes we do fail inside of a function but there's no jump into
    if let Some(location) = &last_instruction.location {
        let failing_function = location.get_containing_function();

        if let Some(failing_function) = failing_function {
            let frame = StackTraceEntry::RevertError {
                source_reference: get_function_start_source_reference(
                    CreateOrCallMessageRef::Call(trace),
                    &failing_function,
                )?,
                return_data: trace.return_data.clone(),
                is_invalid_opcode_error: last_instruction.opcode == OpCode::INVALID,
            };

            return Ok(Heuristic::Hit(vec![frame]));
        }
    }

    let contract = contract_meta.contract.read();

    let selector = calldata.get(..4).unwrap_or(&calldata[..]);
    let calldata = &calldata.get(4..).unwrap_or(&[]);

    let called_function = contract.get_function_from_selector(selector);

    if let Some(called_function) = called_function {
        let abi = alloy_json_abi::Function::try_from(&**called_function)?;

        let is_valid_calldata = match &called_function.param_types {
            Some(_) => abi.abi_decode_input(calldata, true).is_ok(),
            // if we don't know the param types, we just assume that the call is valid
            None => true,
        };

        if !is_valid_calldata {
            let frame = StackTraceEntry::InvalidParamsError {
                source_reference: get_function_start_source_reference(
                    CreateOrCallMessageRef::Call(trace),
                    called_function,
                )?,
            };

            return Ok(Heuristic::Hit(vec![frame]));
        }
    }

    if solidity_0_6_3_maybe_unmapped_revert(CreateOrCallMessageRef::Call(trace))? {
        let revert_frame = solidity_0_6_3_get_frame_for_unmapped_revert_before_function(trace)?;

        if let Some(revert_frame) = revert_frame {
            return Ok(Heuristic::Hit(vec![revert_frame]));
        }
    }

    let frame = get_other_error_before_called_function_stack_trace_entry(trace)?;

    Ok(Heuristic::Hit(vec![frame]))
}

/// Check if the last submessage can be used to generate the stack trace.
fn check_last_submessage<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
    last_submessage_data: Option<SubmessageData<HaltReasonT>>,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    let Some(last_submessage_data) = last_submessage_data else {
        return Ok(Heuristic::Miss(stacktrace));
    };

    let mut inferred_stacktrace = Cow::from(&stacktrace);

    // get the instruction before the submessage and add it to the stack trace
    let call_step = match steps.get(last_submessage_data.step_index as usize - 1) {
        Some(NestedTraceStep::Evm(call_step)) => call_step,
        _ => panic!("This should not happen: MessageTrace should be preceded by a EVM step"),
    };

    let call_inst = contract_meta.get_instruction(call_step.pc)?;
    let call_stack_frame = instruction_to_callstack_stack_trace_entry(&contract_meta, call_inst)?;
    // TODO: remove this expect
    let call_stack_frame_source_reference = call_stack_frame
        .source_reference()
        .cloned()
        .expect("Callstack entry must have source reference");

    let last_message_failed = match &last_submessage_data.message_trace {
        NestedTrace::Create(create) => create.exit.is_error(),
        NestedTrace::Call(call) => call.exit.is_error(),
        NestedTrace::Precompile(precompile) => precompile.exit.is_error(),
    };
    if last_message_failed {
        // add the call/create that generated the message to the stack trace
        let inferred_stacktrace = inferred_stacktrace.to_mut();
        inferred_stacktrace.push(call_stack_frame);

        if is_subtrace_error_propagated(trace, last_submessage_data.step_index)?
            || is_proxy_error_propagated(trace, last_submessage_data.step_index)?
        {
            inferred_stacktrace.extend(last_submessage_data.stacktrace);

            if is_contract_call_run_out_of_gas_error(trace, last_submessage_data.step_index)? {
                let last_frame = match inferred_stacktrace.pop() {
                    Some(frame) => frame,
                    _ => panic!("Expected inferred stack trace to have at least one frame"),
                };

                inferred_stacktrace.push(StackTraceEntry::ContractCallRunOutOfGasError {
                    source_reference: last_frame.source_reference().cloned(),
                });
            }

            return fix_initial_modifier(trace, inferred_stacktrace.to_owned()).map(Heuristic::Hit);
        }
    } else {
        let is_return_data_size_error =
            fails_right_after_call(trace, last_submessage_data.step_index)?;
        if is_return_data_size_error {
            inferred_stacktrace
                .to_mut()
                .push(StackTraceEntry::ReturndataSizeError {
                    source_reference: call_stack_frame_source_reference,
                });

            return fix_initial_modifier(trace, inferred_stacktrace.into_owned())
                .map(Heuristic::Hit);
        }
    }

    Ok(Heuristic::Miss(stacktrace))
}

/// Check if the trace reverted with a panic error.
fn check_panic<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    mut stacktrace: Vec<StackTraceEntry>,
    last_instruction: &Instruction,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    let return_data = ReturnData::new(trace.return_data().clone());

    if !return_data.is_panic_return_data() {
        return Ok(Heuristic::Miss(stacktrace));
    }

    // If the last frame is an internal function, it means that the trace
    // jumped there to return the panic. If that's the case, we remove that
    // frame.
    if let Some(StackTraceEntry::InternalFunctionCallstackEntry { .. }) = stacktrace.last() {
        stacktrace.pop();
    }

    // if the error comes from a call to a zero-initialized function,
    // we remove the last frame, which represents the call, to avoid
    // having duplicated frames
    let error_code = return_data.decode_panic()?;
    if error_code == U256::from(0x51) {
        stacktrace.pop();
    }

    stacktrace.push(instruction_within_function_to_panic_stack_trace_entry(
        trace,
        last_instruction,
        error_code,
    )?);

    fix_initial_modifier(trace, stacktrace).map(Heuristic::Hit)
}

fn check_non_contract_called<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    mut stacktrace: Vec<StackTraceEntry>,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    if is_called_non_contract_account_error(trace)? {
        let source_reference = get_last_source_reference(trace)?;

        // We are sure this is not undefined because there was at least a call
        // instruction
        let source_reference = source_reference.expect("Expected source reference to be defined");

        let non_contract_called_frame =
            StackTraceEntry::NoncontractAccountCalledError { source_reference };

        stacktrace.push(non_contract_called_frame);

        Ok(Heuristic::Hit(stacktrace))
    } else {
        Ok(Heuristic::Miss(stacktrace))
    }
}

/// Check if the execution stopped with a revert or an invalid opcode.
fn check_revert_or_invalid_opcode<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    stacktrace: Vec<StackTraceEntry>,
    last_instruction: &Instruction,
    function_jumpdests: &[&Instruction],
    jumped_into_function: bool,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    match last_instruction.opcode {
        OpCode::REVERT | OpCode::INVALID => {}
        _ => return Ok(Heuristic::Miss(stacktrace)),
    }

    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let return_data = trace.return_data();

    let mut inferred_stacktrace = stacktrace.clone();

    if let Some(location) = &last_instruction.location {
        if jumped_into_function || matches!(trace, CreateOrCallMessageRef::Create(_)) {
            // There should always be a function here, but that's not the case with
            // optimizations.
            //
            // If this is a create trace, we already checked args and nonpayable failures
            // before calling this function.
            //
            // If it's a call trace, we already jumped into a function. But optimizations
            // can happen.
            let failing_function = location.get_containing_function();

            // If the failure is in a modifier we add an entry with the function/constructor
            match failing_function {
                Some(func) if func.r#type == ContractFunctionType::Modifier => {
                    let frame = get_entry_before_failure_in_modifier(trace, function_jumpdests)?;

                    inferred_stacktrace.push(frame);
                }
                _ => {}
            }
        }
    }

    let panic_stacktrace = check_panic(trace, inferred_stacktrace, last_instruction)?;
    let inferred_stacktrace = match panic_stacktrace {
        hit @ Heuristic::Hit(..) => return Ok(hit),
        Heuristic::Miss(stacktrace) => stacktrace,
    };

    let custom_error_stacktrace =
        check_custom_errors(trace, inferred_stacktrace, last_instruction)?;
    let mut inferred_stacktrace = match custom_error_stacktrace {
        hit @ Heuristic::Hit(..) => return Ok(hit),
        Heuristic::Miss(stacktrace) => stacktrace,
    };

    if let Some(location) = &last_instruction.location {
        if jumped_into_function || matches!(trace, CreateOrCallMessageRef::Create(_)) {
            let failing_function = location.get_containing_function();

            if failing_function.is_some() {
                let frame = instruction_within_function_to_revert_stack_trace_entry(
                    trace,
                    last_instruction,
                )?;

                inferred_stacktrace.push(frame);
            } else {
                let is_invalid_opcode_error = last_instruction.opcode == OpCode::INVALID;

                match &trace {
                    CreateOrCallMessageRef::Call(CallMessage { calldata, .. }) => {
                        let contract = contract_meta.contract.read();

                        // This is here because of the optimizations
                        let function_from_selector = contract
                            .get_function_from_selector(calldata.get(..4).unwrap_or(&calldata[..]));

                        // in general this shouldn't happen, but it does when viaIR is enabled,
                        // "optimizerSteps": "u" is used, and the called function is fallback or
                        // receive
                        let Some(function) = function_from_selector else {
                            return Ok(Heuristic::Miss(inferred_stacktrace));
                        };

                        let frame = StackTraceEntry::RevertError {
                            source_reference: get_function_start_source_reference(trace, function)?,
                            return_data: return_data.clone(),
                            is_invalid_opcode_error,
                        };

                        inferred_stacktrace.push(frame);
                    }
                    CreateOrCallMessageRef::Create(create) => {
                        // This is here because of the optimizations
                        let frame = StackTraceEntry::RevertError {
                            source_reference: get_constructor_start_source_reference(create)?,
                            return_data: return_data.clone(),
                            is_invalid_opcode_error,
                        };

                        inferred_stacktrace.push(frame);
                    }
                }
            }

            return fix_initial_modifier(trace, inferred_stacktrace).map(Heuristic::Hit);
        }
    }

    // If the revert instruction is not mapped but there is return data,
    // we add the frame anyway, sith the best sourceReference we can get
    if last_instruction.location.is_none() && !return_data.is_empty() {
        let revert_frame = StackTraceEntry::RevertError {
            source_reference: get_contract_start_without_function_source_reference(trace)?,
            return_data: return_data.clone(),
            is_invalid_opcode_error: last_instruction.opcode == OpCode::INVALID,
        };

        inferred_stacktrace.push(revert_frame);

        return fix_initial_modifier(trace, inferred_stacktrace).map(Heuristic::Hit);
    }

    Ok(Heuristic::Miss(stacktrace))
}

fn check_solidity_0_6_3_unmapped_revert<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    mut stacktrace: Vec<StackTraceEntry>,
) -> Result<Heuristic, InferrerError<HaltReasonT>> {
    if solidity_0_6_3_maybe_unmapped_revert(trace)? {
        let revert_frame = solidity_0_6_3_get_frame_for_unmapped_revert_within_function(trace)?;

        if let Some(revert_frame) = revert_frame {
            stacktrace.push(revert_frame);

            return Ok(Heuristic::Hit(stacktrace));
        }

        return Ok(Heuristic::Hit(stacktrace));
    }

    Ok(Heuristic::Miss(stacktrace))
}

fn empty_calldata_and_no_receive<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let version =
        Version::parse(&contract_meta.compiler_version).expect("Failed to parse SemVer version");

    // this only makes sense when receive functions are available
    if version < FIRST_SOLC_VERSION_RECEIVE_FUNCTION {
        return Ok(false);
    }

    Ok(trace.calldata.is_empty() && contract.receive.is_none())
}

fn fails_right_after_call<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    call_subtrace_step_index: u32,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    let Some(NestedTraceStep::Evm(last_step)) = steps.last() else {
        return Ok(false);
    };

    let last_inst = contract_meta.get_instruction(last_step.pc)?;
    if last_inst.opcode != OpCode::REVERT {
        return Ok(false);
    }

    let call_opcode_step = steps.get(call_subtrace_step_index as usize - 1);
    let call_opcode_step = match call_opcode_step {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => return Err(InferrerError::ExpectedEvmStep),
    };
    let call_inst = contract_meta.get_instruction(call_opcode_step.pc)?;

    // Calls are always made from within functions
    let call_inst_location = call_inst
        .location
        .as_ref()
        .expect("Expected call instruction location to be defined");

    is_last_location(trace, call_subtrace_step_index + 1, call_inst_location)
}

fn fix_initial_modifier<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    mut stacktrace: Vec<StackTraceEntry>,
) -> Result<Vec<StackTraceEntry>, InferrerError<HaltReasonT>> {
    if let Some(StackTraceEntry::CallstackEntry {
        function_type: ContractFunctionType::Modifier,
        ..
    }) = stacktrace.first()
    {
        let entry_before_initial_modifier =
            get_entry_before_initial_modifier_callstack_entry(trace)?;

        stacktrace.insert(0, entry_before_initial_modifier);
    }

    Ok(stacktrace)
}

// Rewrite of `AbiHelpers.formatValues` from Hardhat
fn format_dyn_sol_value(val: &DynSolValue) -> String {
    match val {
        // print nested values as [value1, value2, ...]
        DynSolValue::Array(items)
        | DynSolValue::Tuple(items)
        | DynSolValue::FixedArray(items)
        | DynSolValue::CustomStruct { tuple: items, .. } => {
            let mut result = String::from("[");
            for (i, val) in items.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&format_dyn_sol_value(val));
            }

            result.push(']');
            result
        }
        // surround string values with quotes
        DynSolValue::String(s) => format!("\"{s}\""),

        DynSolValue::Address(address) => format!("\"{address}\""),
        DynSolValue::Bytes(bytes) => format!("\"{}\"", hex::encode_prefixed(bytes)),
        DynSolValue::FixedBytes(word, size) => {
            format!("\"{}\"", hex::encode_prefixed(&word.0.as_slice()[..*size]))
        }
        DynSolValue::Bool(b) => b.to_string(),
        DynSolValue::Function(_) => "<function>".to_string(),
        DynSolValue::Int(int, _bits) => int.to_string(),
        DynSolValue::Uint(uint, _bits) => uint.to_string(),
    }
}

fn get_constructor_start_source_reference<HaltReasonT: HaltReasonTrait>(
    trace: &CreateMessage<HaltReasonT>,
) -> Result<SourceReference, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();
    let contract_location = &contract.location;

    let line = match &contract.constructor {
        Some(constructor) => constructor.location.get_starting_line_number(),
        None => contract_location.get_starting_line_number(),
    };

    let file = contract_location.file();
    let file = file.read();

    Ok(SourceReference {
        source_name: file.source_name.clone(),
        source_content: file.content.clone(),
        contract: Some(contract.name.clone()),
        function: Some(CONSTRUCTOR_FUNCTION_NAME.to_string()),
        line,
        range: (
            contract_location.offset,
            contract_location.offset + contract_location.length,
        ),
    })
}

fn get_contract_start_without_function_source_reference<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<SourceReference, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .clone()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let location = &contract.location;
    let file = location.file();
    let file = file.read();

    Ok(SourceReference {
        source_name: file.source_name.clone(),
        source_content: file.content.clone(),
        contract: Some(contract.name.clone()),

        function: None,
        line: location.get_starting_line_number(),
        range: (location.offset, location.offset + location.length),
    })
}

fn get_direct_library_call_error_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let func =
        contract.get_function_from_selector(trace.calldata.get(..4).unwrap_or(&trace.calldata[..]));

    let source_reference = match func {
        Some(func) => {
            get_function_start_source_reference(CreateOrCallMessageRef::Call(trace), func)?
        }
        None => get_contract_start_without_function_source_reference(
            CreateOrCallMessageRef::Call(trace),
        )?,
    };

    Ok(vec![StackTraceEntry::DirectLibraryCallError {
        source_reference,
    }])
}

fn get_function_start_source_reference<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    func: &ContractFunction,
) -> Result<SourceReference, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let file = func.location.file();
    let file = file.read();

    let location = &func.location;

    Ok(SourceReference {
        source_name: file.source_name.clone(),
        source_content: file.content.clone(),
        contract: Some(contract.name.clone()),

        function: Some(func.name.clone()),
        line: location.get_starting_line_number(),
        range: (location.offset, location.offset + location.length),
    })
}

fn get_entry_before_initial_modifier_callstack_entry<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let trace = match trace {
        CreateOrCallMessageRef::Create(create) => {
            return Ok(StackTraceEntry::CallstackEntry {
                source_reference: get_constructor_start_source_reference(create)?,
                function_type: ContractFunctionType::Constructor,
            })
        }
        CreateOrCallMessageRef::Call(call) => call,
    };

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let called_function =
        contract.get_function_from_selector(trace.calldata.get(..4).unwrap_or(&trace.calldata[..]));

    let source_reference = match called_function {
        Some(called_function) => get_function_start_source_reference(
            CreateOrCallMessageRef::Call(trace),
            called_function,
        )?,
        None => get_fallback_start_source_reference(trace)?,
    };

    let function_type = match called_function {
        Some(_) => ContractFunctionType::Function,
        None => ContractFunctionType::Fallback,
    };

    Ok(StackTraceEntry::CallstackEntry {
        source_reference,
        function_type,
    })
}

fn get_entry_before_failure_in_modifier<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    function_jumpdests: &[&Instruction],
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;

    // If there's a jumpdest, this modifier belongs to the last function that it
    // represents
    if let Some(last_jumpdest) = function_jumpdests.last() {
        let entry = instruction_to_callstack_stack_trace_entry(&contract_meta, last_jumpdest)?;

        return Ok(entry);
    }

    // This function is only called after we jumped into the initial function in
    // call traces, so there should always be at least a function jumpdest.
    let trace = match trace {
        CreateOrCallMessageRef::Call(call) => {
            return Err(InferrerError::MissingFunctionJumpDest(call.clone()))
        }
        CreateOrCallMessageRef::Create(create) => create,
    };

    // If there's no jump dest, we point to the constructor.
    Ok(StackTraceEntry::CallstackEntry {
        source_reference: get_constructor_start_source_reference(trace)?,
        function_type: ContractFunctionType::Constructor,
    })
}

fn get_fallback_start_source_reference<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<SourceReference, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let func = match &contract.fallback {
        Some(func) => func,
        None => panic!("This shouldn't happen: trying to get fallback source reference from a contract without fallback"),
    };

    let location = &func.location;
    let file = location.file();
    let file = file.read();

    Ok(SourceReference {
        source_name: file.source_name.clone(),
        source_content: file.content.clone(),
        contract: Some(contract.name.clone()),
        function: Some(FALLBACK_FUNCTION_NAME.to_string()),
        line: location.get_starting_line_number(),
        range: (location.offset, location.offset + location.length),
    })
}

fn get_last_instruction_with_valid_location_step_index<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Option<u32>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    for (i, step) in steps.iter().enumerate().rev() {
        let step = match step {
            NestedTraceStep::Evm(step) => step,
            _ => return Ok(None),
        };

        let inst = contract_meta.get_instruction(step.pc)?;

        if inst.location.is_some() {
            return Ok(Some(i as u32));
        }
    }

    Ok(None)
}

fn get_last_instruction_with_valid_location<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Option<Instruction>, InferrerError<HaltReasonT>> {
    let last_location_index = get_last_instruction_with_valid_location_step_index(trace)?;

    let Some(last_location_index) = last_location_index else {
        return Ok(None);
    };

    let steps = trace.steps();

    match &steps.get(last_location_index as usize) {
        Some(NestedTraceStep::Evm(step)) => {
            let contract_meta = trace
                .contract_meta()
                .ok_or(InferrerError::MissingContract)?;
            let inst = contract_meta.get_instruction(step.pc)?;

            Ok(Some(inst.clone()))
        }
        _ => Ok(None),
    }
}
fn get_last_source_reference<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Option<SourceReference>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    for step in steps.iter().rev() {
        let step = match step {
            NestedTraceStep::Evm(step) => step,
            _ => continue,
        };

        let inst = contract_meta.get_instruction(step.pc)?;

        let Some(location) = &inst.location else {
            continue;
        };

        let source_reference = source_location_to_source_reference(&contract_meta, Some(location));

        if let Some(source_reference) = source_reference {
            return Ok(Some(source_reference));
        }
    }

    Ok(None)
}

fn get_other_error_before_called_function_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let source_reference =
        get_contract_start_without_function_source_reference(CreateOrCallMessageRef::Call(trace))?;

    Ok(StackTraceEntry::OtherExecutionError {
        source_reference: Some(source_reference),
    })
}

fn has_failed_inside_the_fallback_function<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract = &trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?
        .contract;
    let contract = contract.read();

    match &contract.fallback {
        Some(fallback) => has_failed_inside_function(trace, fallback),
        None => Ok(false),
    }
}

fn has_failed_inside_the_receive_function<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract = &trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?
        .contract;
    let contract = contract.read();

    match &contract.receive {
        Some(receive) => has_failed_inside_function(trace, receive),
        None => Ok(false),
    }
}

fn has_failed_inside_function<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
    func: &ContractFunction,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let last_step = trace
        .steps
        .iter()
        .last()
        .expect("There should at least be one step");

    let last_step = match last_step {
        NestedTraceStep::Evm(step) => step,
        _ => return Err(InferrerError::ExpectedEvmStep),
    };

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;

    let last_instruction = contract_meta.get_instruction(last_step.pc)?;

    Ok(match &last_instruction.location {
        Some(last_instruction_location) => {
            last_instruction.opcode == OpCode::REVERT
                && func.location.contains(last_instruction_location)
        }
        _ => false,
    })
}

fn instruction_within_function_to_custom_error_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    inst: &Instruction,
    message: String,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let last_source_reference = get_last_source_reference(trace)?;
    let last_source_reference =
        last_source_reference.expect("Expected source reference to be defined");

    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;

    let source_reference =
        source_location_to_source_reference(&contract_meta, inst.location.as_deref());

    let source_reference = source_reference.unwrap_or(last_source_reference);

    Ok(StackTraceEntry::CustomError {
        source_reference,
        message,
    })
}

fn instruction_within_function_to_panic_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    inst: &Instruction,
    error_code: U256,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let last_source_reference = get_last_source_reference(trace)?;

    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;

    let source_reference =
        source_location_to_source_reference(&contract_meta, inst.location.as_deref());

    let source_reference = source_reference.or(last_source_reference);

    Ok(StackTraceEntry::PanicError {
        source_reference,
        error_code,
    })
}

fn instruction_within_function_to_revert_stack_trace_entry<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    inst: &Instruction,
) -> Result<StackTraceEntry, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;

    let source_reference =
        source_location_to_source_reference(&contract_meta, inst.location.as_deref())
            .ok_or(InferrerError::MissingSourceReference)?;

    Ok(StackTraceEntry::RevertError {
        source_reference,
        is_invalid_opcode_error: inst.opcode == OpCode::INVALID,
        return_data: trace.return_data().clone(),
    })
}

fn instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference<
    HaltReasonT: HaltReasonTrait,
>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    inst: &Instruction,
) -> Result<Option<SourceReference>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;

    let source_reference =
        source_location_to_source_reference(&contract_meta, inst.location.as_deref());

    Ok(source_reference)
}

fn is_called_non_contract_account_error<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    // We could change this to checking that the last valid location maps to a call,
    // but it's way more complex as we need to get the ast node from that
    // location.

    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    let last_index = get_last_instruction_with_valid_location_step_index(trace)?;

    let last_index = match last_index {
        None | Some(0) => return Ok(false),
        Some(last_index) => last_index as usize,
    };

    let last_step = match &steps[last_index] {
        NestedTraceStep::Evm(step) => step,
        _ => panic!("We know this is an EVM step"),
    };

    let last_inst = contract_meta.get_instruction(last_step.pc)?;

    if last_inst.opcode != OpCode::ISZERO {
        return Ok(false);
    }

    let prev_step = match &steps[last_index - 1] {
        NestedTraceStep::Evm(step) => step,
        _ => panic!("We know this is an EVM step"),
    };

    let prev_inst = contract_meta.get_instruction(prev_step.pc)?;

    Ok(prev_inst.opcode == OpCode::EXTCODESIZE)
}

fn is_call_failed_error<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    inst_index: u32,
    call_instruction: &Instruction,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let call_location = match &call_instruction.location {
        Some(location) => location,
        None => panic!("Expected call location to be defined"),
    };

    is_last_location(trace, inst_index, call_location)
}

/// Returns a source reference pointing to the constructor if it exists, or
/// to the contract otherwise.
fn is_constructor_invalid_arguments_error<HaltReasonT: HaltReasonTrait>(
    trace: &CreateMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    if trace.return_data.len() > 0 {
        return Ok(false);
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    // This function is only matters with contracts that have constructors defined.
    // The ones that don't are abstract contracts, or their constructor
    // doesn't take any argument.
    let Some(constructor) = &contract.constructor else {
        return Ok(false);
    };

    let Ok(version) = Version::parse(&contract_meta.compiler_version) else {
        return Ok(false);
    };
    if version < FIRST_SOLC_VERSION_CREATE_PARAMS_VALIDATION {
        return Ok(false);
    }

    let last_step = trace.steps.last();
    let Some(NestedTraceStep::Evm(last_step)) = last_step else {
        return Ok(false);
    };

    let last_inst = contract_meta.get_instruction(last_step.pc)?;

    if last_inst.opcode != OpCode::REVERT || last_inst.location.is_some() {
        return Ok(false);
    }

    let mut has_read_deployment_code_size = false;
    for step in trace.steps.iter() {
        let step = match step {
            NestedTraceStep::Evm(step) => step,
            _ => return Ok(false),
        };

        let inst = contract_meta.get_instruction(step.pc)?;

        if let Some(inst_location) = &inst.location {
            if contract.location != *inst_location && constructor.location != *inst_location {
                return Ok(false);
            }
        }

        if inst.opcode == OpCode::CODESIZE {
            has_read_deployment_code_size = true;
        }
    }

    Ok(has_read_deployment_code_size)
}

fn is_constructor_not_payable_error<HaltReasonT: HaltReasonTrait>(
    trace: &CreateMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    // This error doesn't return data
    if !trace.return_data.is_empty() {
        return Ok(false);
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    // This function is only matters with contracts that have constructors defined.
    // The ones that don't are abstract contracts, or their constructor
    // doesn't take any argument.
    let constructor = match &contract.constructor {
        Some(constructor) => constructor,
        None => return Ok(false),
    };

    if trace.value.is_zero() {
        return Ok(false);
    }

    Ok(constructor.is_payable != Some(true))
}

fn is_direct_library_call<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract = &trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?
        .contract;
    let contract = contract.read();

    Ok(trace.depth == 0 && contract.r#type == ContractKind::Library)
}

fn is_contract_call_run_out_of_gas_error<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    call_step_index: u32,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let steps = trace.steps();
    let return_data = trace.return_data();
    let exit_code = trace.exit_code();

    if return_data.len() > 0 {
        return Ok(false);
    }

    if !exit_code.is_revert() {
        return Ok(false);
    }

    let call_exit = match steps.get(call_step_index as usize) {
        None | Some(NestedTraceStep::Evm(_)) => panic!("Expected call to be a message trace"),
        Some(NestedTraceStep::Precompile(precompile)) => precompile.exit.clone(),
        Some(NestedTraceStep::Call(call)) => call.exit.clone(),
        Some(NestedTraceStep::Create(create)) => create.exit.clone(),
    };

    if !call_exit.is_out_of_gas_error() {
        return Ok(false);
    }

    fails_right_after_call(trace, call_step_index)
}

fn is_fallback_not_payable_error<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
    called_function: Option<&ContractFunction>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    // This error doesn't return data
    if !trace.return_data.is_empty() {
        return Ok(false);
    }

    if trace.value.is_zero() {
        return Ok(false);
    }

    // the called function exists in the contract
    if called_function.is_some() {
        return Ok(false);
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    match &contract.fallback {
        Some(fallback) => Ok(fallback.is_payable != Some(true)),
        None => Ok(false),
    }
}

fn is_function_not_payable_error<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
    called_function: &ContractFunction,
) -> Result<bool, InferrerError<HaltReasonT>> {
    // This error doesn't return data
    if !trace.return_data.is_empty() {
        return Ok(false);
    }

    if trace.value.is_zero() {
        return Ok(false);
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    // Libraries don't have a nonpayable check
    if contract.r#type == ContractKind::Library {
        return Ok(false);
    }

    Ok(called_function.is_payable != Some(true))
}

fn is_last_location<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    from_step: u32,
    location: &SourceLocation,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    for step in steps.iter().skip(from_step as usize) {
        let step = match step {
            NestedTraceStep::Evm(step) => step,
            _ => return Ok(false),
        };

        let step_inst = contract_meta.get_instruction(step.pc)?;

        if let Some(step_inst_location) = &step_inst.location {
            if **step_inst_location != *location {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

fn is_missing_function_and_fallback_error<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
    called_function: Option<&ContractFunction>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    // This error doesn't return data
    if trace.return_data.len() > 0 {
        return Ok(false);
    }

    // the called function exists in the contract
    if called_function.is_some() {
        return Ok(false);
    }

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    // there's a receive function and no calldata
    if trace.calldata.len() == 0 && contract.receive.is_some() {
        return Ok(false);
    }

    Ok(contract.fallback.is_none())
}

fn is_proxy_error_propagated<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    call_subtrace_step_index: u32,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let trace = match &trace {
        CreateOrCallMessageRef::Call(call) => call,
        CreateOrCallMessageRef::Create(_) => return Ok(false),
    };

    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;

    let call_step = match trace.steps.get(call_subtrace_step_index as usize - 1) {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => return Ok(false),
    };

    let call_inst = contract_meta.get_instruction(call_step.pc)?;

    if call_inst.opcode != OpCode::DELEGATECALL {
        return Ok(false);
    }

    let subtrace = match trace.steps.get(call_subtrace_step_index as usize) {
        None | Some(NestedTraceStep::Evm(_) | NestedTraceStep::Precompile(_)) => return Ok(false),
        Some(NestedTraceStep::Call(call)) => CreateOrCallMessageRef::Call(call),
        Some(NestedTraceStep::Create(create)) => CreateOrCallMessageRef::Create(create),
    };

    let Some(subtrace_contract_meta) = subtrace.contract_meta() else {
        // If we can't recognize the implementation we'd better don't consider it as
        // such
        return Ok(false);
    };

    if subtrace_contract_meta.contract.read().r#type == ContractKind::Library {
        return Ok(false);
    }

    if trace.return_data.as_ref() != subtrace.return_data().as_ref() {
        return Ok(false);
    }

    for step in trace
        .steps
        .iter()
        .skip(call_subtrace_step_index as usize + 1)
    {
        let step = match step {
            NestedTraceStep::Evm(step) => step,
            _ => return Ok(false),
        };

        let inst = contract_meta.get_instruction(step.pc)?;

        // All the remaining locations should be valid, as they are part of the inline
        // asm
        if inst.location.is_none() {
            return Ok(false);
        }

        if matches!(
            inst.jump_type,
            JumpType::IntoFunction | JumpType::OutofFunction
        ) {
            return Ok(false);
        }
    }

    let last_step = match trace.steps.last() {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => return Err(InferrerError::ExpectedEvmStep),
    };
    let last_inst = contract_meta.get_instruction(last_step.pc)?;

    Ok(last_inst.opcode == OpCode::REVERT)
}

fn is_subtrace_error_propagated<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    call_subtrace_step_index: u32,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let return_data = trace.return_data();
    let steps = trace.steps();
    let exit = trace.exit_code();

    let (call_return_data, call_exit) = match steps.get(call_subtrace_step_index as usize) {
        None | Some(NestedTraceStep::Evm(_)) => panic!("Expected call to be a message trace"),
        Some(NestedTraceStep::Precompile(ref precompile)) => {
            (precompile.return_data.clone(), precompile.exit.clone())
        }
        Some(NestedTraceStep::Call(ref call)) => (call.return_data.clone(), call.exit.clone()),
        Some(NestedTraceStep::Create(ref create)) => {
            (create.return_data.clone(), create.exit.clone())
        }
    };

    if return_data.as_ref() != call_return_data.as_ref() {
        return Ok(false);
    }

    if exit.is_out_of_gas_error() && call_exit.is_out_of_gas_error() {
        return Ok(true);
    }

    // If the return data is not empty, and it's still the same, we assume it
    // is being propagated
    if return_data.len() > 0 {
        return Ok(true);
    }

    fails_right_after_call(trace, call_subtrace_step_index)
}

fn other_execution_error_stacktrace<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    mut stacktrace: Vec<StackTraceEntry>,
) -> Result<Vec<StackTraceEntry>, InferrerError<HaltReasonT>> {
    let other_execution_error_frame = StackTraceEntry::OtherExecutionError {
        source_reference: get_last_source_reference(trace)?,
    };

    stacktrace.push(other_execution_error_frame);
    Ok(stacktrace)
}

fn solidity_0_6_3_maybe_unmapped_revert<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();

    if steps.is_empty() {
        return Ok(false);
    }

    let last_step = steps.last();
    let last_step = match last_step {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => return Ok(false),
    };

    let last_instruction = contract_meta.get_instruction(last_step.pc)?;

    let Ok(version) = Version::parse(&contract_meta.compiler_version) else {
        return Ok(false);
    };
    let req = VersionReq::parse(&format!("^{FIRST_SOLC_VERSION_WITH_UNMAPPED_REVERTS}"))
        .expect("valid semver");

    Ok(req.matches(&version) && last_instruction.opcode == OpCode::REVERT)
}

// Solidity 0.6.3 unmapped reverts special handling
// For more info: https://github.com/ethereum/solidity/issues/9006
fn solidity_0_6_3_get_frame_for_unmapped_revert_before_function<HaltReasonT: HaltReasonTrait>(
    trace: &CallMessage<HaltReasonT>,
) -> Result<Option<StackTraceEntry>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta
        .as_ref()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let revert_frame = solidity_0_6_3_get_frame_for_unmapped_revert_within_function(
        CreateOrCallMessageRef::Call(trace),
    )?;

    let revert_frame = match revert_frame {
        None
        | Some(StackTraceEntry::UnmappedSolc0_6_3RevertError {
            source_reference: None,
            ..
        }) => {
            if contract.receive.is_none() || trace.calldata.len() > 0 {
                // Failed within the fallback
                if let Some(fallback) = &contract.fallback {
                    let location = &fallback.location;
                    let file = location.file();
                    let file = file.read();

                    let source_reference = SourceReference {
                        contract: Some(contract.name.clone()),
                        function: Some(FALLBACK_FUNCTION_NAME.to_string()),
                        source_name: file.source_name.clone(),
                        source_content: file.content.clone(),
                        line: location.get_starting_line_number(),
                        range: (location.offset, location.offset + location.length),
                    };
                    let revert_frame = StackTraceEntry::UnmappedSolc0_6_3RevertError {
                        source_reference: Some(solidity_0_6_3_correct_line_number(
                            source_reference,
                        )),
                    };

                    Some(revert_frame)
                } else {
                    None
                }
            } else {
                let receive = contract
                    .receive
                    .as_ref()
                    .expect("None always hits branch above");

                let location = &receive.location;
                let file = location.file();
                let file = file.read();

                let source_reference = SourceReference {
                    contract: Some(contract.name.clone()),
                    function: Some(RECEIVE_FUNCTION_NAME.to_string()),
                    source_name: file.source_name.clone(),
                    source_content: file.content.clone(),
                    line: location.get_starting_line_number(),
                    range: (location.offset, location.offset + location.length),
                };
                let revert_frame = StackTraceEntry::UnmappedSolc0_6_3RevertError {
                    source_reference: Some(solidity_0_6_3_correct_line_number(source_reference)),
                };

                Some(revert_frame)
            }
        }
        Some(revert_frame) => Some(revert_frame),
    };

    Ok(revert_frame)
}

fn solidity_0_6_3_get_frame_for_unmapped_revert_within_function<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Option<StackTraceEntry>, InferrerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let contract = contract_meta.contract.read();

    let steps = trace.steps();

    // If we are within a function there's a last valid location. It may
    // be the entire contract.
    let prev_inst = get_last_instruction_with_valid_location(trace)?;
    let last_step = match steps.last() {
        Some(NestedTraceStep::Evm(step)) => step,
        _ => return Err(InferrerError::ExpectedEvmStep),
    };
    let next_inst_pc = last_step.pc + 1;
    let has_next_inst = contract_meta.has_instruction(next_inst_pc);

    if has_next_inst {
        let next_inst = contract_meta.get_instruction(next_inst_pc)?;

        let prev_loc = prev_inst.as_ref().and_then(|i| i.location.as_deref());
        let next_loc = next_inst.location.as_deref();

        let prev_func = prev_loc.and_then(SourceLocation::get_containing_function);
        let next_func = next_loc.and_then(SourceLocation::get_containing_function);

        // This is probably a require. This means that we have the exact
        // line, but the stack trace may be degraded (e.g. missing our
        // synthetic call frames when failing in a modifier) so we still
        // add this frame as UNMAPPED_SOLC_0_6_3_REVERT_ERROR
        match (&prev_func, &next_loc, &prev_loc) {
            (Some(_), Some(next_loc), Some(prev_loc)) if prev_loc == next_loc => {
                let source_reference = instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference(
                    trace,
                    next_inst,
                )?;
                return Ok(Some(StackTraceEntry::UnmappedSolc0_6_3RevertError {
                    source_reference,
                }));
            }
            _ => {}
        }

        let source_reference = if prev_func.is_some() && prev_inst.is_some() {
            instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference(
                trace,
                prev_inst.as_ref().unwrap(),
            )?
        } else if next_func.is_some() {
            instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference(
                trace, next_inst,
            )?
        } else {
            None
        };

        return Ok(Some(StackTraceEntry::UnmappedSolc0_6_3RevertError {
            source_reference: source_reference.map(solidity_0_6_3_correct_line_number),
        }));
    }

    if matches!(trace, CreateOrCallMessageRef::Create(_)) && prev_inst.is_some() {
        // Solidity is smart enough to stop emitting extra instructions after
        // an unconditional revert happens in a constructor. If this is the case
        // we just return a special error.

        let source_reference =
            instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference(
                trace,
                prev_inst.as_ref().unwrap(),
            )?
            .map_or_else(
                || {
                    // When the latest instruction is not within a function we need
                    // some default sourceReference to show to the user
                    let location = &contract.location;
                    let file = location.file();
                    let file = file.read();

                    let mut default_source_reference = SourceReference {
                        function: Some(CONSTRUCTOR_FUNCTION_NAME.to_string()),
                        contract: Some(contract.name.clone()),
                        source_name: file.source_name.clone(),
                        source_content: file.content.clone(),
                        line: location.get_starting_line_number(),
                        range: (location.offset, location.offset + location.length),
                    };

                    if let Some(constructor) = &contract.constructor {
                        default_source_reference.line =
                            constructor.location.get_starting_line_number();
                    }

                    default_source_reference
                },
                solidity_0_6_3_correct_line_number,
            );

        return Ok(Some(StackTraceEntry::UnmappedSolc0_6_3RevertError {
            source_reference: Some(source_reference),
        }));
    }

    if let Some(prev_inst) = prev_inst {
        // We may as well just be in a function or modifier and just happen
        // to be at the last instruction of the runtime bytecode.
        // In this case we just return whatever the last mapped intruction
        // points to.
        let source_reference =
            instruction_within_function_to_unmapped_solc_0_6_3_revert_error_source_reference(
                trace, &prev_inst,
            )?
            .map(solidity_0_6_3_correct_line_number);

        return Ok(Some(StackTraceEntry::UnmappedSolc0_6_3RevertError {
            source_reference,
        }));
    }

    Ok(None)
}

fn solidity_0_6_3_correct_line_number(mut source_reference: SourceReference) -> SourceReference {
    let lines: Vec<_> = source_reference.source_content.split('\n').collect();

    let current_line = lines[source_reference.line as usize - 1];
    if current_line.contains("require") || current_line.contains("revert") {
        return source_reference;
    }

    let next_lines = &lines
        .get(source_reference.line as usize..)
        .unwrap_or_default();
    let first_non_empty_line = next_lines.iter().position(|l| !l.trim().is_empty());

    let Some(first_non_empty_line) = first_non_empty_line else {
        return source_reference;
    };

    let next_line = next_lines[first_non_empty_line];
    if next_line.contains("require") || next_line.contains("revert") {
        source_reference.line += 1 + first_non_empty_line as u32;
    }

    source_reference
}

fn source_location_to_source_reference(
    contract_meta: &ContractMetadata,
    location: Option<&SourceLocation>,
) -> Option<SourceReference> {
    let location = location?;
    let func = location.get_containing_function()?;

    let func_name = match func.r#type {
        ContractFunctionType::Constructor => CONSTRUCTOR_FUNCTION_NAME.to_string(),
        ContractFunctionType::Fallback => FALLBACK_FUNCTION_NAME.to_string(),
        ContractFunctionType::Receive => RECEIVE_FUNCTION_NAME.to_string(),
        _ => func.name.clone(),
    };

    let func_location_file = func.location.file();
    let func_location_file = func_location_file.read();

    Some(SourceReference {
        function: Some(func_name.clone()),
        contract: if func.r#type == ContractFunctionType::FreeFunction {
            None
        } else {
            Some(contract_meta.contract.read().name.clone())
        },
        source_name: func_location_file.source_name.clone(),
        source_content: func_location_file.content.clone(),
        line: location.get_starting_line_number(),
        range: (location.offset, location.offset + location.length),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sol_value_to_string() {
        assert_eq!(
            format_dyn_sol_value(&DynSolValue::String("hello".to_string())),
            "\"hello\""
        );
        // Uniform, 0-prefixed hex strings
        assert_eq!(
            format_dyn_sol_value(&DynSolValue::Address([0u8; 20].into())),
            format!(r#""0x{}""#, "0".repeat(2 * 20))
        );
        assert_eq!(
            format_dyn_sol_value(&DynSolValue::Bytes(vec![0u8; 32])),
            format!(r#""0x{}""#, "0".repeat(2 * 32))
        );
        assert_eq!(
            format_dyn_sol_value(&DynSolValue::FixedBytes([0u8; 32].into(), 10)),
            format!(r#""0x{}""#, "0".repeat(2 * 10))
        );
        assert_eq!(
            format_dyn_sol_value(&DynSolValue::FixedBytes([0u8; 32].into(), 32)),
            format!(r#""0x{}""#, "0".repeat(2 * 32))
        );
    }
}
