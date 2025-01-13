//! Naive Rust port of the `VmTracer` from Hardhat.

use std::{cell::RefCell, rc::Rc};

use edr_eth::{result::ExecutionResult, spec::HaltReasonTrait, Address, Bytes, U160, U256};
use edr_evm::trace::{BeforeMessage, Step};

use crate::{
    exit_code::ExitCode,
    nested_trace::{
        CallMessage, CreateMessage, EvmStep, NestedTrace, NestedTraceStep, PrecompileMessage,
    },
};

/// Errors that can occur during the generation of the nested trace.
#[derive(Debug, thiserror::Error)]
pub enum NestedTracerError {
    /// Invalid input: The created address should be defined in the successful
    #[error("Created address should be defined in successful create trace")]
    MissingAddressInExecutionResult,
    /// Invalid input: Missing code address.
    #[error("Missing code address")]
    MissingCodeAddress,
    /// Invalid input: Missing code.
    #[error("Missing code")]
    MissingCode,
    /// Invalid input: Message execution started while a precompile was
    /// executing.
    #[error("Message execution started while a precompile was executing")]
    MessageDuringPreCompile,
    /// Invalid input: Step event fired while a precompile was executing.
    #[error("Step event fired while a precompile was executing")]
    StepDuringPreCompile,
}

/// Observes a trace, collecting information about the execution of the EVM.
pub fn convert_trace_messages_to_nested_trace<HaltReasonT: HaltReasonTrait>(
    trace: edr_evm::trace::Trace<HaltReasonT>,
) -> Result<Option<NestedTrace<HaltReasonT>>, NestedTracerError> {
    let mut tracer = NestedTracer::new();

    tracer.add_messages(trace.messages)?;

    Ok(tracer.get_last_top_level_message_trace())
}

/// Naive Rust port of the `VmTracer` from Hardhat.
struct NestedTracer<HaltReasonT: HaltReasonTrait> {
    tracing_steps: Vec<Step>,
    message_traces: Vec<Rc<RefCell<InternalNestedTrace<HaltReasonT>>>>,
}

impl<HaltReasonT: HaltReasonTrait> Default for NestedTracer<HaltReasonT> {
    fn default() -> Self {
        Self::new()
    }
}

// Temporarily hardcoded to remove the need of using ethereumjs' common and evm
// TODO(#565): We should be using a more robust check by checking the hardfork
// config (and potentially other config like optional support for RIP
// precompiles, which start at 0x100).
const MAX_PRECOMPILE_NUMBER: u16 = 10;

impl<HaltReasonT: HaltReasonTrait> NestedTracer<HaltReasonT> {
    /// Creates a new [`NestedTracer`].
    const fn new() -> Self {
        NestedTracer {
            tracing_steps: Vec::new(),
            message_traces: Vec::new(),
        }
    }

    /// Returns a reference to the last top-level message trace.
    fn get_last_top_level_message_trace(mut self) -> Option<NestedTrace<HaltReasonT>> {
        self.message_traces.pop().map(convert_to_external_trace)
    }

    fn add_messages(
        &mut self,
        messages: Vec<edr_evm::trace::TraceMessage<HaltReasonT>>,
    ) -> Result<(), NestedTracerError> {
        for msg in messages {
            match msg {
                edr_evm::trace::TraceMessage::Before(before) => {
                    self.add_before_message(before)?;
                }
                edr_evm::trace::TraceMessage::Step(step) => {
                    self.add_step(step)?;
                }
                edr_evm::trace::TraceMessage::After(after) => {
                    self.add_after_message(after.execution_result)?;
                }
            }
        }
        Ok(())
    }

    fn add_before_message(&mut self, message: BeforeMessage) -> Result<(), NestedTracerError> {
        let trace: InternalNestedTrace<HaltReasonT>;

        if message.depth == 0 {
            self.message_traces.clear();
            self.tracing_steps.clear();
        }

        if let Some(to) = message.to {
            let to_as_u160 = U160::from_be_bytes(**to);

            if to_as_u160 <= U160::from(MAX_PRECOMPILE_NUMBER) {
                let precompile: u32 = to_as_u160
                    .try_into()
                    .expect("MAX_PRECOMPILE_NUMBER is of type u16 so it fits");

                let precompile_trace = PrecompileMessage {
                    value: message.value,
                    exit: ExitCode::Success,
                    return_data: Bytes::new(),
                    depth: message.depth,
                    gas_used: 0,
                    precompile,
                    calldata: message.data,
                };

                trace = InternalNestedTrace::Precompile(precompile_trace);
            } else {
                // if we enter here, then `to` is not None, therefore
                // `code_address` and `code` should be Some
                let code_address = message
                    .code_address
                    .ok_or(NestedTracerError::MissingCodeAddress)?;
                let code = message.code.ok_or(NestedTracerError::MissingCode)?;

                let call_trace = InternalCallMessage {
                    steps: Vec::new(),
                    calldata: message.data,
                    address: to,
                    code_address,
                    depth: message.depth,
                    value: message.value,
                    exit: ExitCode::Success,
                    return_data: Bytes::new(),
                    gas_used: 0,
                    code: code.original_bytes(),
                    number_of_subtraces: 0,
                };

                trace = InternalNestedTrace::Call(call_trace);
            }
        } else {
            let create_trace = InternalCreateMessage {
                number_of_subtraces: 0,
                steps: Vec::new(),
                depth: message.depth,
                value: message.value,
                exit: ExitCode::Success,
                return_data: Bytes::new(),
                gas_used: 0,
                code: message.data,
                deployed_contract: None,
            };

            trace = InternalNestedTrace::Create(create_trace);
        }

        // We need to share it so that adding steps when processing via stack
        // also updates the inner elements
        let trace = Rc::new(RefCell::new(trace));

        if let Some(parent_ref) = self.message_traces.last_mut() {
            let mut parent_trace = parent_ref.borrow_mut();
            match &mut *parent_trace {
                InternalNestedTrace::Precompile(_) => {
                    return Err(NestedTracerError::MessageDuringPreCompile);
                }
                InternalNestedTrace::Create(create) => {
                    create
                        .steps
                        .push(InternalNestedTraceStep::Message(Rc::clone(&trace)));
                    create.number_of_subtraces += 1;
                }
                InternalNestedTrace::Call(call) => {
                    call.steps
                        .push(InternalNestedTraceStep::Message(Rc::clone(&trace)));
                    call.number_of_subtraces += 1;
                }
            };
        }

        self.message_traces.push(trace);

        Ok(())
    }

    fn add_step(&mut self, step: Step) -> Result<(), NestedTracerError> {
        if let Some(parent_ref) = self.message_traces.last_mut() {
            let mut parent_trace = parent_ref.borrow_mut();
            let steps = match &mut *parent_trace {
                InternalNestedTrace::Precompile(_) => {
                    return Err(NestedTracerError::StepDuringPreCompile);
                }
                InternalNestedTrace::Create(create) => &mut create.steps,
                InternalNestedTrace::Call(call) => &mut call.steps,
            };

            steps.push(InternalNestedTraceStep::Evm(EvmStep { pc: step.pc }));
        }

        self.tracing_steps.push(step);

        Ok(())
    }

    fn add_after_message(
        &mut self,
        result: ExecutionResult<HaltReasonT>,
    ) -> Result<(), NestedTracerError> {
        if let Some(trace) = self.message_traces.last_mut() {
            let mut trace = trace.borrow_mut();

            trace.set_gas_used(result.gas_used());

            match result {
                ExecutionResult::Success { output, .. } => {
                    trace.set_exit_code(ExitCode::Success);
                    trace.set_return_data(output.data().clone());

                    if let InternalNestedTrace::Create(trace) = &mut *trace {
                        let address = output
                            .address()
                            .ok_or(NestedTracerError::MissingAddressInExecutionResult)?;

                        trace.deployed_contract = Some(address.as_slice().to_vec().into());
                    }
                }
                ExecutionResult::Halt { reason, .. } => {
                    trace.set_exit_code(ExitCode::Halt(reason));
                    trace.set_return_data(Bytes::new());
                }
                ExecutionResult::Revert { output, .. } => {
                    trace.set_exit_code(ExitCode::Revert);
                    trace.set_return_data(output);
                }
            }
        }

        if self.message_traces.len() > 1 {
            self.message_traces.pop();
        }

        Ok(())
    }
}

/// A nested trace where the message steps are shared and mutable via a
/// refcell.
#[derive(Clone, Debug)]
enum InternalNestedTrace<HaltReasonT: HaltReasonTrait> {
    Create(InternalCreateMessage<HaltReasonT>),
    Call(InternalCallMessage<HaltReasonT>),
    Precompile(PrecompileMessage<HaltReasonT>),
}

impl<HaltReasonT: HaltReasonTrait> InternalNestedTrace<HaltReasonT> {
    fn set_gas_used(&mut self, gas_used: u64) {
        match self {
            InternalNestedTrace::Create(create) => create.gas_used = gas_used,
            InternalNestedTrace::Call(call) => call.gas_used = gas_used,

            InternalNestedTrace::Precompile(precompile) => precompile.gas_used = gas_used,
        }
    }

    fn set_exit_code(&mut self, exit_code: ExitCode<HaltReasonT>) {
        match self {
            InternalNestedTrace::Create(create) => create.exit = exit_code,
            InternalNestedTrace::Call(call) => call.exit = exit_code,

            InternalNestedTrace::Precompile(precompile) => precompile.exit = exit_code,
        }
    }

    fn set_return_data(&mut self, return_data: Bytes) {
        match self {
            InternalNestedTrace::Create(create) => create.return_data = return_data,
            InternalNestedTrace::Call(call) => call.return_data = return_data,

            InternalNestedTrace::Precompile(precompile) => precompile.return_data = return_data,
        }
    }
}

/// Represents a call message.
#[derive(Clone, Debug)]
struct InternalCallMessage<HaltReasonT: HaltReasonTrait> {
    // The following is just an optimization: When processing this traces it's useful to know ahead
    // of time how many subtraces there are.
    /// Number of subtraces. Used to speed up the processing of the traces in
    /// JS.
    pub number_of_subtraces: u32,
    /// Children messages.
    pub steps: Vec<InternalNestedTraceStep<HaltReasonT>>,
    /// Calldata buffer
    pub calldata: Bytes,
    /// Address of the contract that is being executed.
    pub address: Address,
    /// Address of the code that is being executed.
    pub code_address: Address,
    /// Code of the contract that is being executed.
    pub code: Bytes,
    /// Value of the message.
    pub value: U256,
    /// Return data buffer.
    pub return_data: Bytes,
    /// EVM exit code.
    pub exit: ExitCode<HaltReasonT>,
    /// How much gas was used.
    pub gas_used: u64,
    /// Depth of the message.
    pub depth: usize,
}

/// Represents a create message.
#[derive(Clone, Debug)]
struct InternalCreateMessage<HaltReasonT: HaltReasonTrait> {
    // The following is just an optimization: When processing this traces it's useful to know ahead
    // of time how many subtraces there are.
    /// Number of subtraces. Used to speed up the processing of the traces in
    /// JS.
    pub number_of_subtraces: u32,
    /// Children messages.
    pub steps: Vec<InternalNestedTraceStep<HaltReasonT>>,
    /// Address of the deployed contract.
    pub deployed_contract: Option<Bytes>,
    /// Code of the contract that is being executed.
    pub code: Bytes,
    /// Value of the message.
    pub value: U256,
    /// Return data buffer.
    pub return_data: Bytes,
    /// EVM exit code.
    pub exit: ExitCode<HaltReasonT>,
    /// How much gas was used.
    pub gas_used: u64,
    /// Depth of the message.
    pub depth: usize,
}

/// Represents a message step. Naive Rust port of the `MessageTraceStep`
/// from Hardhat.
#[derive(Clone, Debug)]
enum InternalNestedTraceStep<HaltReasonT: HaltReasonTrait> {
    /// [`NestedTrace`] variant.
    // It's both read and written to (updated) by the `[NestedTracer]`.
    Message(Rc<RefCell<InternalNestedTrace<HaltReasonT>>>),
    /// [`EvmStep`] variant.
    Evm(EvmStep),
}

enum InternalNestedTraceStepWithoutRefCell<HaltReasonT: HaltReasonTrait> {
    Message(NestedTrace<HaltReasonT>),
    Evm(EvmStep),
}

/// Converts the [`InternalNestedTrace`] into a [`NestedTrace`] by
/// cloning it.
///
/// # Panics
///
///  Panics if the value is mutably borrowed.
fn convert_to_external_trace<HaltReasonT: HaltReasonTrait>(
    value: Rc<RefCell<InternalNestedTrace<HaltReasonT>>>,
) -> NestedTrace<HaltReasonT> {
    // We can't use `Rc::try_unwrap` because it requires that the `Rc` is unique.
    let trace = value.borrow().clone();

    match trace {
        InternalNestedTrace::Create(create) => {
            let InternalCreateMessage {
                number_of_subtraces,
                steps,
                deployed_contract,
                code,
                value,
                return_data,
                exit,
                gas_used,
                depth,
            } = create;

            NestedTrace::Create(CreateMessage {
                number_of_subtraces,
                steps: steps.into_iter().map(convert_to_external_step).collect(),
                contract_meta: None,
                deployed_contract,
                code,
                value,
                return_data,
                exit,
                gas_used,
                depth,
            })
        }
        InternalNestedTrace::Call(call) => {
            let InternalCallMessage {
                number_of_subtraces,
                steps,
                calldata,
                address,
                code_address,
                code,
                value,
                return_data,
                exit,
                gas_used,
                depth,
            } = call;
            NestedTrace::Call(CallMessage {
                number_of_subtraces,
                steps: steps.into_iter().map(convert_to_external_step).collect(),
                contract_meta: None,
                calldata,
                address,
                code_address,
                code,
                value,
                return_data,
                exit,
                gas_used,
                depth,
            })
        }
        InternalNestedTrace::Precompile(precompile) => NestedTrace::Precompile(precompile),
    }
}

/// # Panics
//  Panics if a nested value is mutably borrowed.
fn convert_to_external_step<HaltReasonT: HaltReasonTrait>(
    value: InternalNestedTraceStep<HaltReasonT>,
) -> NestedTraceStep<HaltReasonT> {
    match value {
        InternalNestedTraceStep::Message(message) => {
            InternalNestedTraceStepWithoutRefCell::Message(convert_to_external_trace(message))
        }
        InternalNestedTraceStep::Evm(evm_step) => {
            InternalNestedTraceStepWithoutRefCell::Evm(evm_step)
        }
    }
    .into()
}

// This can be a `From` conversion, because it can't panic.
impl<HaltReasonT: HaltReasonTrait> From<InternalNestedTraceStepWithoutRefCell<HaltReasonT>>
    for NestedTraceStep<HaltReasonT>
{
    fn from(step: InternalNestedTraceStepWithoutRefCell<HaltReasonT>) -> Self {
        match step {
            InternalNestedTraceStepWithoutRefCell::Message(trace) => match trace {
                NestedTrace::Create(create_trace) => NestedTraceStep::Create(create_trace),
                NestedTrace::Call(call_trace) => NestedTraceStep::Call(call_trace),
                NestedTrace::Precompile(precompile_trace) => {
                    NestedTraceStep::Precompile(precompile_trace)
                }
            },
            InternalNestedTraceStepWithoutRefCell::Evm(evm_step) => NestedTraceStep::Evm(evm_step),
        }
    }
}
