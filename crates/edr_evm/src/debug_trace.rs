use std::{collections::HashMap, fmt::Debug, sync::Arc};

use edr_eth::{
    block::Block as _,
    bytecode::opcode::{self, OpCode},
    hex, l1,
    result::{ExecutionResult, InvalidTransaction, ResultAndState},
    spec::{ChainSpec, HaltReasonTrait},
    transaction::{ExecutableTransaction as _, TransactionValidation},
    utils::u256_to_padded_hex,
    Address, Bytes, B256, U256,
};
use revm::Evm;

use crate::{
    blockchain::SyncBlockchain,
    config::{CfgEnv, Env},
    debug::GetContextData,
    evm::{
        handler::register::EvmHandler,
        interpreter::{table::DynInstruction, Interpreter, InterpreterResult},
        Context, EvmWiring, InnerEvmContext, JournalEntry,
    },
    spec::RuntimeSpec,
    state::{Database, DatabaseComponents, SyncState, WrapDatabaseRef},
    trace::{register_trace_collector_handles, Trace, TraceCollector},
    transaction::TransactionError,
};

/// EIP-3155 and raw tracers.
pub struct Eip3155AndRawTracers<HaltReasonT: HaltReasonTrait> {
    eip3155: TracerEip3155,
    raw: TraceCollector<HaltReasonT>,
}

impl<HaltReasonT: HaltReasonTrait> Eip3155AndRawTracers<HaltReasonT> {
    /// Creates a new instance.
    pub fn new(config: DebugTraceConfig, verbose_tracing: bool) -> Self {
        Self {
            eip3155: TracerEip3155::new(config),
            raw: TraceCollector::new(verbose_tracing),
        }
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<TraceCollector<HaltReasonT>>
    for Eip3155AndRawTracers<HaltReasonT>
{
    fn get_context_data(&mut self) -> &mut TraceCollector<HaltReasonT> {
        &mut self.raw
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<TracerEip3155>
    for Eip3155AndRawTracers<HaltReasonT>
{
    fn get_context_data(&mut self) -> &mut TracerEip3155 {
        &mut self.eip3155
    }
}

/// Register EIP-3155 and trace collector handles.
pub fn register_eip_3155_and_raw_tracers_handles<
    EvmWiringT: revm::EvmWiring<
        ExternalContext: GetContextData<TraceCollector<EvmWiringT::HaltReason>>
                             + GetContextData<TracerEip3155>,
        Database: Database<Error: Debug>,
    >,
>(
    handler: &mut EvmHandler<'_, EvmWiringT>,
) {
    register_trace_collector_handles(handler);
    register_eip_3155_tracer_handles(handler);
}

/// Get trace output for `debug_traceTransaction`
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
#[allow(clippy::too_many_arguments)]
pub fn debug_trace_transaction<ChainSpecT, BlockchainErrorT, StateErrorT>(
    blockchain: &dyn SyncBlockchain<ChainSpecT, BlockchainErrorT, StateErrorT>,
    // Take ownership of the state so that we can apply throw-away modifications on it
    mut state: Box<dyn SyncState<StateErrorT>>,
    evm_config: CfgEnv,
    hardfork: ChainSpecT::Hardfork,
    trace_config: DebugTraceConfig,
    block: ChainSpecT::BlockEnv,
    transactions: Vec<ChainSpecT::SignedTransaction>,
    transaction_hash: &B256,
    verbose_tracing: bool,
) -> Result<
    DebugTraceResultWithTraces<ChainSpecT::HaltReason>,
    DebugTraceError<ChainSpecT, BlockchainErrorT, StateErrorT>,
>
where
    ChainSpecT: RuntimeSpec<
        BlockEnv: Clone,
        SignedTransaction: Default
                               + TransactionValidation<ValidationError: From<InvalidTransaction>>,
    >,
    BlockchainErrorT: Debug + Send,
    StateErrorT: Debug + Send,
{
    let evm_spec_id = hardfork.into();
    if evm_spec_id < l1::SpecId::SPURIOUS_DRAGON {
        // Matching Hardhat Network behaviour: https://github.com/NomicFoundation/hardhat/blob/af7e4ce6a18601ec9cd6d4aa335fa7e24450e638/packages/hardhat-core/src/internal/hardhat-network/provider/vm/ethereumjs.ts#L427
        return Err(DebugTraceError::InvalidSpecId {
            spec_id: evm_spec_id,
        });
    }

    for transaction in transactions {
        if transaction.transaction_hash() == transaction_hash {
            let mut tracer = Eip3155AndRawTracers::new(trace_config, verbose_tracing);

            let ResultAndState { result, .. } = {
                let env = Env::boxed(evm_config, block, transaction);

                let mut evm = Evm::<ChainSpecT::EvmWiring<_, _>>::builder()
                    .with_db(WrapDatabaseRef(DatabaseComponents {
                        blockchain,
                        state: state.as_ref(),
                    }))
                    .with_external_context(&mut tracer)
                    .with_env(env)
                    .with_spec_id(hardfork)
                    .append_handler_register(register_eip_3155_and_raw_tracers_handles)
                    .build();

                evm.transact().map_err(TransactionError::from)?
            };

            return Ok(execution_result_to_debug_result(result, tracer));
        } else {
            let ResultAndState { state: changes, .. } = {
                let env = Env::boxed(evm_config.clone(), block.clone(), transaction);

                let mut evm = Evm::<ChainSpecT::EvmWiring<_, ()>>::builder()
                    .with_db(WrapDatabaseRef(DatabaseComponents {
                        blockchain,
                        state: state.as_ref(),
                    }))
                    .with_external_context(())
                    .with_env(env)
                    .with_spec_id(hardfork)
                    .build();

                evm.transact().map_err(TransactionError::from)?
            };

            state.commit(changes);
        }
    }

    Err(DebugTraceError::InvalidTransactionHash {
        transaction_hash: *transaction_hash,
        block_number: *block.number(),
    })
}

/// Convert an `ExecutionResult` to a `DebugTraceResult`.
pub fn execution_result_to_debug_result<HaltReasonT: HaltReasonTrait>(
    execution_result: ExecutionResult<HaltReasonT>,
    tracer: Eip3155AndRawTracers<HaltReasonT>,
) -> DebugTraceResultWithTraces<HaltReasonT> {
    let Eip3155AndRawTracers { eip3155, raw } = tracer;
    let traces = raw.into_traces();

    let result = match execution_result {
        ExecutionResult::Success {
            gas_used, output, ..
        } => DebugTraceResult {
            pass: true,
            gas_used,
            output: Some(output.into_data()),
            logs: eip3155.logs,
        },
        ExecutionResult::Revert { gas_used, output } => DebugTraceResult {
            pass: false,
            gas_used,
            output: Some(output),
            logs: eip3155.logs,
        },
        ExecutionResult::Halt { gas_used, .. } => DebugTraceResult {
            pass: false,
            gas_used,
            output: None,
            logs: eip3155.logs,
        },
    };

    DebugTraceResultWithTraces { result, traces }
}

/// Config options for `debug_traceTransaction`
#[derive(Debug, Default, Clone)]
pub struct DebugTraceConfig {
    /// Disable storage trace.
    pub disable_storage: bool,
    /// Disable memory trace.
    pub disable_memory: bool,
    /// Disable stack trace.
    pub disable_stack: bool,
}

/// Debug trace error.
#[derive(Debug, thiserror::Error)]
pub enum DebugTraceError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: ChainSpec,
{
    /// Invalid hardfork spec argument.
    #[error("Invalid spec id: {spec_id:?}. `debug_traceTransaction` is not supported prior to Spurious Dragon")]
    InvalidSpecId {
        /// The hardfork.
        spec_id: l1::SpecId,
    },
    /// Invalid transaction hash argument.
    #[error("Transaction hash {transaction_hash} not found in block {block_number}")]
    InvalidTransactionHash {
        /// The transaction hash.
        transaction_hash: B256,
        /// The block number.
        block_number: U256,
    },
    /// Transaction error.
    #[error(transparent)]
    TransactionError(#[from] TransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>),
}

/// Result of a `debug_traceTransaction` call.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugTraceResult {
    /// Whether transaction was executed successfully.
    pub pass: bool,
    /// All gas used by the transaction.
    pub gas_used: u64,
    /// Return values of the function.
    pub output: Option<Bytes>,
    /// The EIP-3155 debug logs.
    #[serde(rename = "structLogs")]
    pub logs: Vec<DebugTraceLogItem>,
}

/// Result of a `debug_traceTransaction` call with traces.
pub struct DebugTraceResultWithTraces<HaltReasonT: HaltReasonTrait> {
    /// The result of the transaction.
    pub result: DebugTraceResult,
    /// The raw traces of the debugged transaction.
    pub traces: Vec<Trace<HaltReasonT>>,
}

/// The output of an EIP-3155 trace.
/// The required fields match <https://eips.ethereum.org/EIPS/eip-3155#output> except for
/// `returnData` and `refund` which are not used currently by Hardhat.
/// The `opName`, `error`, `memory` and `storage` optional fields are supported
/// as well.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugTraceLogItem {
    /// Program Counter
    pub pc: u64,
    /// Op code
    pub op: u8,
    /// Gas left before executing this operation as hex number.
    pub gas: String,
    /// Gas cost of this operation as hex number.
    pub gas_cost: String,
    /// Array of all values (hex numbers) on the stack
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<Vec<String>>,
    /// Depth of the call stack
    pub depth: u64,
    /// Size of memory array.
    pub mem_size: u64,
    /// Name of the operation.
    pub op_name: String,
    /// Description of an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Array of all allocated values as hex strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<Vec<String>>,
    /// Map of all stored values with keys and values encoded as hex strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<HashMap<String, String>>,
}

/// Register EIP-3155 tracer handles.
pub fn register_eip_3155_tracer_handles<
    EvmWiringT: revm::EvmWiring<ExternalContext: GetContextData<TracerEip3155>>,
>(
    handler: &mut EvmHandler<'_, EvmWiringT>,
) {
    let table = &mut handler.instruction_table;

    // Update all instructions to call the instruction handler.
    table.update_all(instruction_handler);

    // call outcome
    let prev_handle = handler.execution.insert_call_outcome.clone();
    handler.execution.insert_call_outcome = Arc::new(move |ctx, frame, shared_memory, outcome| {
        let tracer = ctx.external.get_context_data();
        tracer.on_inner_frame_result(&outcome.result);

        prev_handle(ctx, frame, shared_memory, outcome)
    });

    // create outcome
    let prev_handle = handler.execution.insert_create_outcome.clone();
    handler.execution.insert_create_outcome = Arc::new(move |ctx, frame, outcome| {
        let tracer = ctx.external.get_context_data();
        tracer.on_inner_frame_result(&outcome.result);

        prev_handle(ctx, frame, outcome)
    });
}

/// Outer closure that calls tracer for every instruction.
fn instruction_handler<EvmWiringT: EvmWiring<ExternalContext: GetContextData<TracerEip3155>>>(
    prev: &DynInstruction<'_, Context<EvmWiringT>>,
    interpreter: &mut Interpreter,
    host: &mut Context<EvmWiringT>,
) {
    // SAFETY: as the PC was already incremented we need to subtract 1 to preserve
    // the old Inspector behavior.
    interpreter.instruction_pointer = unsafe { interpreter.instruction_pointer.sub(1) };

    host.external.get_context_data().step(interpreter);

    // Reset PC to previous value.
    interpreter.instruction_pointer = unsafe { interpreter.instruction_pointer.add(1) };

    // Execute instruction.
    prev(interpreter, host);

    host.external
        .get_context_data()
        .step_end(interpreter, &host.evm);
}

/// An EIP-3155 compatible EVM tracer.
#[derive(Debug)]
pub struct TracerEip3155 {
    config: DebugTraceConfig,
    logs: Vec<DebugTraceLogItem>,
    contract_address: Address,
    gas_remaining: u64,
    memory: Vec<u8>,
    mem_size: usize,
    opcode: u8,
    pc: usize,
    stack: Vec<U256>,
    // Contract-specific storage
    storage: HashMap<Address, HashMap<String, String>>,
}

impl TracerEip3155 {
    /// Create a new tracer.
    pub fn new(config: DebugTraceConfig) -> Self {
        Self {
            config,
            logs: Vec::default(),
            contract_address: Address::default(),
            stack: Vec::new(),
            pc: 0,
            opcode: 0,
            gas_remaining: 0,
            memory: Vec::default(),
            mem_size: 0,
            storage: HashMap::default(),
        }
    }

    fn step(&mut self, interp: &Interpreter) {
        self.contract_address = interp.contract.target_address;
        self.gas_remaining = interp.gas().remaining();

        if !self.config.disable_stack {
            self.stack.clone_from(interp.stack.data());
        }

        if !self.config.disable_memory {
            self.memory = interp.shared_memory.context_memory().to_vec();
        }

        self.mem_size = interp.shared_memory.context_memory().len();

        self.opcode = interp.current_opcode();

        self.pc = interp.program_counter();
    }

    fn step_end<EvmWiringT: EvmWiring>(
        &mut self,
        interp: &Interpreter,
        context: &InnerEvmContext<EvmWiringT>,
    ) {
        let depth = context.journaled_state.depth();

        let stack = if self.config.disable_stack {
            None
        } else {
            Some(
                self.stack
                    .iter()
                    .map(u256_to_padded_hex)
                    .collect::<Vec<String>>(),
            )
        };

        let memory = if self.config.disable_memory {
            None
        } else {
            Some(self.memory.chunks(32).map(hex::encode).collect())
        };

        let storage = if self.config.disable_storage {
            None
        } else {
            if matches!(self.opcode, opcode::SLOAD | opcode::SSTORE) {
                let last_entry = context
                    .journaled_state
                    .journal
                    .last()
                    .and_then(|v| v.last());

                if let Some(
                    JournalEntry::StorageChanged { address, key, .. }
                    | JournalEntry::StorageWarmed { address, key },
                ) = last_entry
                {
                    let value = context.journaled_state.state[address].storage[key].present_value();
                    let contract_storage = self.storage.entry(self.contract_address).or_default();
                    contract_storage.insert(u256_to_padded_hex(key), u256_to_padded_hex(&value));
                }
            }
            Some(
                self.storage
                    .get(&self.contract_address)
                    .cloned()
                    .unwrap_or_default(),
            )
        };

        let mut error = None;
        let op_name = OpCode::new(self.opcode).map_or_else(
            || {
                // Matches message from Hardhat
                // https://github.com/NomicFoundation/hardhat/blob/37c5c5845969b15995cc96cb6bd0596977f8b1f8/packages/hardhat-core/src/internal/hardhat-network/stack-traces/vm-debug-tracer.ts#L452
                let fallback = format!("opcode 0x${:x} not defined", self.opcode);
                error = Some(fallback.clone());
                fallback
            },
            |opcode| opcode.to_string(),
        );

        let gas_cost = self.gas_remaining.saturating_sub(interp.gas().remaining());
        let log_item = DebugTraceLogItem {
            pc: self.pc as u64,
            op: self.opcode,
            gas: format!("0x{:x}", self.gas_remaining),
            gas_cost: format!("0x{gas_cost:x}"),
            stack,
            depth,
            mem_size: self.mem_size as u64,
            op_name,
            error,
            memory,
            storage,
        };
        self.logs.push(log_item);
    }

    fn on_inner_frame_result(&mut self, result: &InterpreterResult) {
        self.gas_remaining = if result.result.is_error() {
            0
        } else {
            result.gas.remaining()
        };
    }
}

impl GetContextData<TracerEip3155> for TracerEip3155 {
    fn get_context_data(&mut self) -> &mut TracerEip3155 {
        self
    }
}
