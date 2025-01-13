use std::{cell::RefCell, fmt::Debug, rc::Rc, sync::Arc};

use derive_where::derive_where;
use edr_eth::{
    bytecode::opcode,
    result::{EVMErrorWiring, ExecutionResult, Output},
    spec::HaltReasonTrait,
    Address, Bytecode, Bytes, U256,
};

use crate::{
    debug::GetContextData,
    evm::{
        handler::register::EvmHandler,
        interpreter::{
            return_revert, table::DynInstruction, CallInputs, CallOutcome, CallValue, CreateInputs,
            CreateOutcome, Interpreter, SuccessOrHalt,
        },
        Context, FrameOrResult, FrameResult, InnerEvmContext,
    },
    spec::EvmWiring,
    state::Database,
};

/// Registers trace collector handles to the EVM handler.
pub fn register_trace_collector_handles<
    EvmWiringT: revm::EvmWiring<
        ExternalContext: GetContextData<TraceCollector<EvmWiringT::HaltReason>>,
        Database: Database<Error: Debug>,
    >,
>(
    handler: &mut EvmHandler<'_, EvmWiringT>,
) {
    let table = &mut handler.instruction_table;

    // Update all instructions to call the instruction handler.
    table.update_all(instruction_handler);

    // call and create input stack shared between handlers. They are used to share
    // inputs in *_end Inspector calls.
    let call_input_stack = Rc::<RefCell<Vec<_>>>::new(RefCell::new(Vec::new()));
    let create_input_stack = Rc::<RefCell<Vec<_>>>::new(RefCell::new(Vec::new()));

    // Create handler
    let create_input_stack_inner = create_input_stack.clone();
    let old_handle = handler.execution.create.clone();
    handler.execution.create = Arc::new(
        move |ctx, inputs| -> Result<FrameOrResult, EVMErrorWiring<EvmWiringT>> {
            let tracer = ctx.external.get_context_data();
            tracer.create(&ctx.evm, &inputs);

            create_input_stack_inner.borrow_mut().push(inputs.clone());

            old_handle(ctx, inputs)
        },
    );

    // Call handler
    let call_input_stack_inner = call_input_stack.clone();
    let old_handle = handler.execution.call.clone();
    handler.execution.call = Arc::new(
        move |ctx, inputs| -> Result<FrameOrResult, EVMErrorWiring<EvmWiringT>> {
            let tracer = ctx.external.get_context_data();
            tracer.call(&mut ctx.evm, &inputs);

            call_input_stack_inner.borrow_mut().push(inputs.clone());

            old_handle(ctx, inputs)
        },
    );

    // call outcome
    let call_input_stack_inner = call_input_stack.clone();
    let old_handle = handler.execution.insert_call_outcome.clone();
    handler.execution.insert_call_outcome = Arc::new(
        move |ctx: &mut revm::Context<EvmWiringT>, frame, shared_memory, outcome| {
            let call_inputs = call_input_stack_inner.borrow_mut().pop().unwrap();

            let tracer = ctx.external.get_context_data();
            tracer.call_end(&ctx.evm, &call_inputs, &outcome);

            old_handle(ctx, frame, shared_memory, outcome)
        },
    );

    // create outcome
    let create_input_stack_inner = create_input_stack.clone();
    let old_handle = handler.execution.insert_create_outcome.clone();
    handler.execution.insert_create_outcome = Arc::new(move |ctx, frame, outcome| {
        let create_inputs = create_input_stack_inner.borrow_mut().pop().unwrap();

        let tracer = ctx.external.get_context_data();
        tracer.create_end(&ctx.evm, &create_inputs, &outcome);

        old_handle(ctx, frame, outcome)
    });

    // last frame outcome
    let old_handle = handler.execution.last_frame_return.clone();
    handler.execution.last_frame_return = Arc::new(move |ctx, frame_result| {
        let tracer = ctx.external.get_context_data();
        match frame_result {
            FrameResult::Call(outcome) => {
                let call_inputs = call_input_stack.borrow_mut().pop().unwrap();
                tracer.call_transaction_end(&ctx.evm, &call_inputs, outcome);
            }
            FrameResult::Create(outcome) => {
                let create_inputs = create_input_stack.borrow_mut().pop().unwrap();
                tracer.create_transaction_end(&ctx.evm, &create_inputs, outcome);
            }
            // TODO: https://github.com/NomicFoundation/edr/issues/427
            FrameResult::EOFCreate(_) => {
                unreachable!("EDR doesn't support EOF yet.")
            }
        }
        old_handle(ctx, frame_result)
    });
}

/// Outer closure that calls tracer for every instruction.
fn instruction_handler<EvmWiringT>(
    prev: &DynInstruction<'_, Context<EvmWiringT>>,
    interpreter: &mut Interpreter,
    host: &mut Context<EvmWiringT>,
) where
    EvmWiringT: revm::EvmWiring<
        Database: Database<Error: Debug>,
        ExternalContext: GetContextData<TraceCollector<EvmWiringT::HaltReason>>,
    >,
{
    // SAFETY: as the PC was already incremented we need to subtract 1 to preserve
    // the old Inspector behavior.
    interpreter.instruction_pointer = unsafe { interpreter.instruction_pointer.sub(1) };

    host.external
        .get_context_data()
        .step(interpreter, &host.evm);

    // Reset PC to previous value.
    interpreter.instruction_pointer = unsafe { interpreter.instruction_pointer.add(1) };

    // Execute instruction.
    prev(interpreter, host);
}

/// Stack tracing message
#[derive(Clone, Debug)]
pub enum TraceMessage<HaltReasonT: HaltReasonTrait> {
    /// Event that occurs before a call or create message.
    Before(BeforeMessage),
    /// Event that occurs every step of a call or create message.
    Step(Step),
    /// Event that occurs after a call or create message.
    After(AfterMessage<HaltReasonT>),
}

/// Temporary before message type for handling traces
#[derive(Clone, Debug)]
pub struct BeforeMessage {
    /// Call depth
    pub depth: usize,
    /// Caller
    pub caller: Address,
    /// Callee
    pub to: Option<Address>,
    /// Whether the call is a static call
    pub is_static_call: bool,
    /// Transaction gas limit
    pub gas_limit: u64,
    /// Input data
    pub data: Bytes,
    /// Value
    pub value: U256,
    /// Code address
    pub code_address: Option<Address>,
    /// Bytecode
    pub code: Option<Bytecode>,
}

/// Event that occurs after a call or create message.
#[derive(Clone, Debug)]
pub struct AfterMessage<HaltReasonT: HaltReasonTrait> {
    /// The execution result
    pub execution_result: ExecutionResult<HaltReasonT>,
    /// The newly created contract address if it's a create tx. `None`
    /// if there was an error creating the contract.
    pub contract_address: Option<Address>,
}

/// A trace for an EVM call.
#[derive(Clone, Debug)]
#[derive_where(Default)]
pub struct Trace<HaltReasonT: HaltReasonTrait> {
    // /// The individual steps of the call
    // pub steps: Vec<Step>,
    /// Messages
    pub messages: Vec<TraceMessage<HaltReasonT>>,
    /// The return value of the call
    pub return_value: Bytes,
}

/// A single EVM step.
#[derive(Clone, Debug)]
pub struct Step {
    /// The program counter
    pub pc: u32,
    /// The call depth
    pub depth: u64,
    /// The executed op code
    pub opcode: u8,
    /// `Stack::Full` if verbose tracing is enabled, `Stack::Top` otherwise
    pub stack: Stack,
    /// Array of all allocated values. Only present if verbose tracing is
    /// enabled.
    pub memory: Option<Vec<u8>>,
    // /// The amount of gas that was used by the step
    // pub gas_cost: u64,
    // /// The amount of gas that was refunded by the step
    // pub gas_refunded: i64,
    // /// The contract being executed
    // pub contract: AccountInfo,
    // /// The address of the contract
    // pub contract_address: Address,
}

/// The stack at a step.
#[derive(Clone, Debug)]
pub enum Stack {
    /// The top of the stack at a step. None if the stack is empty.
    Top(Option<U256>),
    /// The full stack at a step.
    Full(Vec<U256>),
}

impl Stack {
    /// Get the top of the stack.
    pub fn top(&self) -> Option<&U256> {
        match self {
            Stack::Top(top) => top.as_ref(),
            Stack::Full(stack) => stack.last(),
        }
    }

    /// Get the full stack if it has been recorded.
    pub fn full(&self) -> Option<&Vec<U256>> {
        match self {
            Stack::Top(_) => None,
            Stack::Full(stack) => Some(stack),
        }
    }
}

impl<HaltReasonT: HaltReasonTrait> Trace<HaltReasonT> {
    /// Adds a before message
    pub fn add_before(&mut self, message: BeforeMessage) {
        self.messages.push(TraceMessage::Before(message));
    }

    /// Adds a result message
    pub fn add_after(&mut self, message: AfterMessage<HaltReasonT>) {
        self.messages.push(TraceMessage::After(message));
    }

    /// Adds a VM step to the trace
    pub fn add_step(&mut self, step: Step) {
        self.messages.push(TraceMessage::Step(step));
    }
}

/// Object that gathers trace information during EVM execution and can be turned
/// into a trace upon completion.
#[derive(Debug)]
pub struct TraceCollector<HaltReasonT: HaltReasonTrait> {
    traces: Vec<Trace<HaltReasonT>>,
    pending_before: Option<BeforeMessage>,
    is_new_trace: bool,
    verbose: bool,
}

impl<HaltReasonT: HaltReasonTrait> TraceCollector<HaltReasonT> {
    /// Create a trace collector. If verbose is `true` full stack and memory
    /// will be recorded.
    pub fn new(verbose: bool) -> Self {
        Self {
            traces: Vec::new(),
            pending_before: None,
            is_new_trace: true,
            verbose,
        }
    }

    /// Converts the [`TraceCollector`] into its [`Trace`].
    pub fn into_traces(self) -> Vec<Trace<HaltReasonT>> {
        self.traces
    }

    /// Returns the traces collected so far.
    pub fn traces(&self) -> &[Trace<HaltReasonT>] {
        &self.traces
    }

    fn current_trace_mut(&mut self) -> &mut Trace<HaltReasonT> {
        self.traces.last_mut().expect("Trace must have been added")
    }

    fn validate_before_message(&mut self) {
        if let Some(message) = self.pending_before.take() {
            self.current_trace_mut().add_before(message);
        }
    }

    fn call<EvmWiringT: EvmWiring<HaltReason = HaltReasonT, Database: Database<Error: Debug>>>(
        &mut self,
        data: &mut InnerEvmContext<EvmWiringT>,
        inputs: &CallInputs,
    ) {
        if self.is_new_trace {
            self.is_new_trace = false;
            self.traces.push(Trace::default());
        }

        self.validate_before_message();

        // This needs to be split into two functions to avoid borrow checker issues
        #[allow(clippy::map_unwrap_or)]
        let code = data
            .journaled_state
            .state
            .get(&inputs.bytecode_address)
            .map(|account| account.info.clone())
            .map(|mut account_info| {
                if let Some(code) = account_info.code.take() {
                    code
                } else {
                    data.db.code_by_hash(account_info.code_hash).unwrap()
                }
            })
            .unwrap_or_else(|| {
                data.db.basic(inputs.bytecode_address).unwrap().map_or(
                    // If an invalid contract address was provided, return empty code
                    Bytecode::new(),
                    |account_info| {
                        account_info.code.unwrap_or_else(|| {
                            data.db.code_by_hash(account_info.code_hash).unwrap()
                        })
                    },
                )
            });

        self.pending_before = Some(BeforeMessage {
            depth: data.journaled_state.depth,
            caller: inputs.caller,
            to: Some(inputs.target_address),
            is_static_call: inputs.is_static,
            gas_limit: inputs.gas_limit,
            data: inputs.input.clone(),
            value: match inputs.value {
                CallValue::Transfer(value) | CallValue::Apparent(value) => value,
            },
            code_address: Some(inputs.bytecode_address),
            code: Some(code),
        });
    }

    fn call_end<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        data: &InnerEvmContext<EvmWiringT>,
        _inputs: &CallInputs,
        outcome: &CallOutcome,
    ) {
        // TODO: Replace this with the `return_revert!` macro
        use crate::evm::interpreter::InstructionResult;

        match outcome.instruction_result() {
            return_revert!() if self.pending_before.is_some() => {
                self.pending_before = None;
                return;
            }
            _ => (),
        }

        self.validate_before_message();

        let ret = *outcome.instruction_result();
        let safe_ret = if ret == InstructionResult::CallTooDeep
            || ret == InstructionResult::OutOfFunds
            || ret == InstructionResult::StateChangeDuringStaticCall
        {
            InstructionResult::Revert
        } else {
            ret
        };

        let execution_result = match SuccessOrHalt::from(safe_ret) {
            SuccessOrHalt::Success(reason) => ExecutionResult::Success {
                reason,
                gas_used: outcome.gas().spent(),
                gas_refunded: outcome.gas().refunded() as u64,
                logs: data.journaled_state.logs.clone(),
                output: Output::Call(outcome.output().clone()),
            },
            SuccessOrHalt::Revert => ExecutionResult::Revert {
                gas_used: outcome.gas().spent(),
                output: outcome.output().clone(),
            },
            SuccessOrHalt::Halt(reason) => ExecutionResult::Halt {
                reason,
                gas_used: outcome.gas().limit(),
            },
            SuccessOrHalt::Internal(_) => {
                panic!("Internal error: {safe_ret:?}")
            }
            SuccessOrHalt::FatalExternalError => panic!("Fatal external error"),
        };

        self.current_trace_mut().add_after(AfterMessage {
            execution_result,
            contract_address: None,
        });
    }

    fn create<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        data: &InnerEvmContext<EvmWiringT>,
        inputs: &CreateInputs,
    ) {
        if self.is_new_trace {
            self.is_new_trace = false;
            self.traces.push(Trace::default());
        }

        self.validate_before_message();

        self.pending_before = Some(BeforeMessage {
            depth: data.journaled_state.depth,
            caller: inputs.caller,
            to: None,
            gas_limit: inputs.gas_limit,
            is_static_call: false,
            data: inputs.init_code.clone(),
            value: inputs.value,
            code_address: None,
            code: None,
        });
    }

    fn create_end<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        data: &InnerEvmContext<EvmWiringT>,
        _inputs: &CreateInputs,
        outcome: &CreateOutcome,
    ) {
        // TODO: Replace this with the `return_revert!` macro
        use crate::evm::interpreter::InstructionResult;

        self.validate_before_message();

        let ret = *outcome.instruction_result();
        let safe_ret =
            if ret == InstructionResult::CallTooDeep || ret == InstructionResult::OutOfFunds {
                InstructionResult::Revert
            } else {
                ret
            };

        let execution_result = match SuccessOrHalt::from(safe_ret) {
            SuccessOrHalt::Success(reason) => ExecutionResult::Success {
                reason,
                gas_used: outcome.gas().spent(),
                gas_refunded: outcome.gas().refunded() as u64,
                logs: data.journaled_state.logs.clone(),
                output: Output::Create(outcome.output().clone(), outcome.address),
            },
            SuccessOrHalt::Revert => ExecutionResult::Revert {
                gas_used: outcome.gas().spent(),
                output: outcome.output().clone(),
            },
            SuccessOrHalt::Halt(reason) => ExecutionResult::Halt {
                reason,
                gas_used: outcome.gas().limit(),
            },
            SuccessOrHalt::Internal(_) => {
                panic!("Internal error: {safe_ret:?}")
            }
            SuccessOrHalt::FatalExternalError => panic!("Fatal external error"),
        };

        self.current_trace_mut().add_after(AfterMessage {
            execution_result,
            contract_address: outcome.address,
        });
    }

    fn step<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        interp: &Interpreter,
        data: &InnerEvmContext<EvmWiringT>,
    ) {
        // Skip the step
        let skip_step = self.pending_before.as_ref().map_or(false, |message| {
            message.code.is_some() && interp.current_opcode() == opcode::STOP
        });

        self.validate_before_message();

        if !skip_step {
            let stack = if self.verbose {
                Stack::Full(interp.stack.data().clone())
            } else {
                Stack::Top(interp.stack.data().last().cloned())
            };
            let memory = if self.verbose {
                Some(interp.shared_memory.context_memory().to_vec())
            } else {
                None
            };
            self.current_trace_mut().add_step(Step {
                pc: interp
                    .program_counter()
                    .try_into()
                    .expect("program counter fits into u32"),
                depth: data.journaled_state.depth(),
                opcode: interp.current_opcode(),
                stack,
                memory,
            });
        }
    }

    fn call_transaction_end<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        data: &InnerEvmContext<EvmWiringT>,
        inputs: &CallInputs,
        outcome: &CallOutcome,
    ) {
        self.is_new_trace = true;
        self.call_end(data, inputs, outcome);
    }

    fn create_transaction_end<EvmWiringT: EvmWiring<HaltReason = HaltReasonT>>(
        &mut self,
        data: &InnerEvmContext<EvmWiringT>,
        inputs: &CreateInputs,
        outcome: &CreateOutcome,
    ) {
        self.is_new_trace = true;
        self.create_end(data, inputs, outcome);
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<TraceCollector<HaltReasonT>>
    for TraceCollector<HaltReasonT>
{
    fn get_context_data(&mut self) -> &mut TraceCollector<HaltReasonT> {
        self
    }
}
