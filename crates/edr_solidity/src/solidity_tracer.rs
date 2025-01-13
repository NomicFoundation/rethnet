//! Generates JS-style stack traces for Solidity errors.

use edr_eth::{bytecode::opcode::OpCode, spec::HaltReasonTrait};

use crate::{
    build_model::{ContractMetadataError, Instruction, JumpType},
    error_inferrer,
    error_inferrer::{InferrerError, SubmessageData},
    mapped_inline_internal_functions_heuristics::{
        adjust_stack_trace, stack_trace_may_require_adjustments, HeuristicsError,
    },
    nested_trace::{
        CallMessage, CreateMessage, CreateOrCallMessage, CreateOrCallMessageRef, EvmStep,
        NestedTrace, NestedTraceStep, PrecompileMessage,
    },
    solidity_stack_trace::StackTraceEntry,
};

/// Errors that can occur during the generation of the stack trace.
#[derive(Debug, thiserror::Error)]
pub enum SolidityTracerError<HaltReasonT: HaltReasonTrait> {
    /// Errors that can occur when decoding the contract metadata.
    #[error(transparent)]
    ContractMetadata(#[from] ContractMetadataError),
    /// Errors that can occur during the inference of the stack trace.
    #[error(transparent)]
    ErrorInferrer(#[from] InferrerError<HaltReasonT>),
    /// Errors that can occur during the heuristics.
    #[error(transparent)]
    Heuristics(#[from] HeuristicsError),
}

/// Generates a stack trace for the provided nested trace.
pub fn get_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: NestedTrace<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    if !trace.exit_code().is_error() {
        return Ok(Vec::default());
    }

    match trace {
        NestedTrace::Precompile(precompile) => Ok(get_precompile_message_stack_trace(&precompile)?),
        NestedTrace::Call(call) if call.contract_meta.is_some() => {
            Ok(get_call_message_stack_trace(call)?)
        }
        NestedTrace::Create(create) if create.contract_meta.is_some() => {
            Ok(get_create_message_stack_trace(create)?)
        }
        // No bytecode is present
        NestedTrace::Call(ref call) => Ok(get_unrecognized_message_stack_trace(
            CreateOrCallMessageRef::Call(call),
        )?),
        NestedTrace::Create(ref create) => Ok(get_unrecognized_message_stack_trace(
            CreateOrCallMessageRef::Create(create),
        )?),
    }
}

fn get_last_subtrace<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Option<NestedTrace<HaltReasonT>> {
    if trace.number_of_subtraces() == 0 {
        return None;
    }

    trace
        .steps()
        .iter()
        .cloned()
        .rev()
        .find_map(|step| match step {
            NestedTraceStep::Evm(EvmStep { .. }) => None,
            NestedTraceStep::Precompile(precompile) => Some(NestedTrace::Precompile(precompile)),
            NestedTraceStep::Call(call) => Some(NestedTrace::Call(call)),
            NestedTraceStep::Create(create) => Some(NestedTrace::Create(create)),
        })
}

fn get_precompile_message_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: &PrecompileMessage<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    Ok(vec![StackTraceEntry::PrecompileError {
        precompile: trace.precompile,
    }])
}

fn get_create_message_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: CreateMessage<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    let inferred_error = error_inferrer::infer_before_tracing_create_message(&trace)?;

    if let Some(inferred_error) = inferred_error {
        return Ok(inferred_error);
    }

    trace_evm_execution(trace.into())
}

fn get_call_message_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: CallMessage<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    let inferred_error = error_inferrer::infer_before_tracing_call_message(&trace)?;

    if let Some(inferred_error) = inferred_error {
        return Ok(inferred_error);
    }

    trace_evm_execution(trace.into())
}

fn get_unrecognized_message_stack_trace<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    let trace_exit_kind = trace.exit_code();
    let trace_return_data = trace.return_data();

    let subtrace = get_last_subtrace(trace);

    if let Some(subtrace) = subtrace {
        let (is_error, return_data) = match &subtrace {
            NestedTrace::Precompile(precompile) => {
                (precompile.exit.is_error(), precompile.return_data.clone())
            }
            NestedTrace::Call(call) => (call.exit.is_error(), call.return_data.clone()),
            NestedTrace::Create(create) => (create.exit.is_error(), create.return_data.clone()),
        };

        // This is not a very exact heuristic, but most of the time it will be right, as
        // solidity reverts if a call fails, and most contracts are in
        // solidity
        if is_error && trace_return_data.as_ref() == return_data.as_ref() {
            let unrecognized_entry: StackTraceEntry = match trace {
                CreateOrCallMessageRef::Call(CallMessage { address, .. }) => {
                    StackTraceEntry::UnrecognizedContractCallstackEntry { address: *address }
                }
                CreateOrCallMessageRef::Create(_) => {
                    StackTraceEntry::UnrecognizedCreateCallstackEntry
                }
            };

            let mut stacktrace = vec![unrecognized_entry];
            stacktrace.extend(get_stack_trace(subtrace)?);

            return Ok(stacktrace);
        }
    }

    if trace_exit_kind.is_contract_too_large_error() {
        return Ok(vec![StackTraceEntry::ContractTooLargeError {
            source_reference: None,
        }]);
    }

    let is_invalid_opcode_error = trace_exit_kind.is_invalid_opcode_error();

    match trace {
        CreateOrCallMessageRef::Call(call) => {
            Ok(vec![StackTraceEntry::UnrecognizedContractError {
                address: call.address,
                return_data: call.return_data.clone(),
                is_invalid_opcode_error,
            }])
        }
        CreateOrCallMessageRef::Create(call) => {
            Ok(vec![StackTraceEntry::UnrecognizedCreateError {
                return_data: call.return_data.clone(),
                is_invalid_opcode_error,
            }])
        }
    }
}

fn trace_evm_execution<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessage<HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    let stack_trace = raw_trace_evm_execution(CreateOrCallMessageRef::from(&trace))?;

    if stack_trace_may_require_adjustments(&stack_trace, CreateOrCallMessageRef::from(&trace))? {
        return adjust_stack_trace(stack_trace, CreateOrCallMessageRef::from(&trace))
            .map_err(SolidityTracerError::from);
    }

    Ok(stack_trace)
}

fn raw_trace_evm_execution<HaltReasonT: HaltReasonTrait>(
    trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, SolidityTracerError<HaltReasonT>> {
    let contract_meta = trace
        .contract_meta()
        .ok_or(InferrerError::MissingContract)?;
    let steps = trace.steps();
    let number_of_subtraces = trace.number_of_subtraces();

    let mut stacktrace: Vec<StackTraceEntry> = Vec::default();

    let mut subtraces_seen = 0;

    // There was a jump into a function according to the sourcemaps
    let mut jumped_into_function = false;

    let mut function_jumpdests: Vec<&Instruction> = Vec::default();

    let mut last_submessage_data: Option<SubmessageData<HaltReasonT>> = None;

    let mut iter = steps.iter().enumerate().peekable();
    while let Some((step_index, step)) = iter.next() {
        if let NestedTraceStep::Evm(EvmStep { pc }) = step {
            let inst = contract_meta.get_instruction(*pc)?;

            if inst.jump_type == JumpType::IntoFunction && iter.peek().is_some() {
                let (_, next_step) = iter.peek().unwrap();
                let NestedTraceStep::Evm(next_evm_step) = next_step else {
                    return Err(InferrerError::ExpectedEvmStep.into());
                };
                let next_inst = contract_meta.get_instruction(next_evm_step.pc)?;

                if next_inst.opcode == OpCode::JUMPDEST {
                    let frame = error_inferrer::instruction_to_callstack_stack_trace_entry(
                        &contract_meta,
                        inst,
                    )?;
                    stacktrace.push(frame);
                    if next_inst.location.is_some() {
                        jumped_into_function = true;
                    }
                    function_jumpdests.push(next_inst);
                }
            } else if inst.jump_type == JumpType::OutofFunction {
                stacktrace.pop();
                function_jumpdests.pop();
            }
        } else {
            let message_trace = match step {
                NestedTraceStep::Evm(_) => unreachable!("branch is taken above"),
                NestedTraceStep::Precompile(precompile) => {
                    NestedTrace::Precompile(precompile.clone())
                }
                NestedTraceStep::Call(call) => NestedTrace::Call(call.clone()),
                NestedTraceStep::Create(create) => NestedTrace::Create(create.clone()),
            };

            subtraces_seen += 1;

            // If there are more subtraces, this one didn't terminate the execution
            if subtraces_seen < number_of_subtraces {
                continue;
            }

            let submessage_trace = get_stack_trace(message_trace.clone())?;

            last_submessage_data = Some(SubmessageData {
                message_trace,
                step_index: step_index as u32,
                stacktrace: submessage_trace,
            });
        }
    }

    let stacktrace_with_inferred_error = error_inferrer::infer_after_tracing(
        trace,
        stacktrace,
        &function_jumpdests,
        jumped_into_function,
        last_submessage_data,
    )?;

    error_inferrer::filter_redundant_frames(stacktrace_with_inferred_error)
        .map_err(SolidityTracerError::from)
}
