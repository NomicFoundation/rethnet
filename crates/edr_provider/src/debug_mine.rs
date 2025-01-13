use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc};

use edr_eth::{
    result::ExecutionResult,
    spec::{ChainSpec, HaltReasonTrait},
    transaction::ExecutableTransaction,
    Bytes, B256,
};
use edr_evm::{
    spec::RuntimeSpec,
    state::{StateDiff, SyncState},
    trace::Trace,
    Block, MineBlockResultAndState,
};

/// The result of mining a block, including the state, in debug mode. This
/// result needs to be inserted into the blockchain to be persistent.
pub struct DebugMineBlockResultAndState<HaltReasonT: HaltReasonTrait, LocalBlockT, StateErrorT> {
    /// Mined block
    pub block: LocalBlockT,
    /// State after mining the block
    pub state: Box<dyn SyncState<StateErrorT>>,
    /// State diff applied by block
    pub state_diff: StateDiff,
    /// Transaction results
    pub transaction_results: Vec<ExecutionResult<HaltReasonT>>,
    /// Transaction traces
    pub transaction_traces: Vec<Trace<HaltReasonT>>,
    /// Encoded `console.log` call inputs
    pub console_log_inputs: Vec<Bytes>,
}

impl<HaltReasonT: HaltReasonTrait, LocalBlockT, StateErrorT>
    DebugMineBlockResultAndState<HaltReasonT, LocalBlockT, StateErrorT>
{
    /// Constructs a new instance from a [`MineBlockResultAndState`],
    /// transaction traces, and decoded console log messages.
    pub fn new(
        result: MineBlockResultAndState<HaltReasonT, LocalBlockT, StateErrorT>,
        transaction_traces: Vec<Trace<HaltReasonT>>,
        console_log_decoded_messages: Vec<Bytes>,
    ) -> Self {
        Self {
            block: result.block,
            state: result.state,
            state_diff: result.state_diff,
            transaction_results: result.transaction_results,
            transaction_traces,
            console_log_inputs: console_log_decoded_messages,
        }
    }
}

/// Helper type for a chain-specific [`DebugMineBlockResult`].
pub type DebugMineBlockResultForChainSpec<ChainSpecT> = DebugMineBlockResult<
    Arc<<ChainSpecT as RuntimeSpec>::Block>,
    <ChainSpecT as ChainSpec>::HaltReason,
    <ChainSpecT as ChainSpec>::SignedTransaction,
>;

/// The result of mining a block in debug mode, after having been committed to
/// the blockchain.
#[derive(Clone, Debug)]
pub struct DebugMineBlockResult<BlockT, HaltReasonT: HaltReasonTrait, SignedTransactionT> {
    /// Mined block
    pub block: BlockT,
    /// Transaction results
    pub transaction_results: Vec<ExecutionResult<HaltReasonT>>,
    /// Transaction traces
    pub transaction_traces: Vec<Trace<HaltReasonT>>,
    /// Encoded `console.log` call inputs
    pub console_log_inputs: Vec<Bytes>,
    phantom: PhantomData<SignedTransactionT>,
}

impl<BlockT, HaltReasonT: HaltReasonTrait, SignedTransactionT>
    DebugMineBlockResult<BlockT, HaltReasonT, SignedTransactionT>
{
    /// Constructs a new instance.
    pub fn new(
        block: BlockT,
        transaction_results: Vec<ExecutionResult<HaltReasonT>>,
        transaction_traces: Vec<Trace<HaltReasonT>>,
        console_log_inputs: Vec<Bytes>,
    ) -> Self {
        Self {
            block,
            transaction_results,
            transaction_traces,
            console_log_inputs,
            phantom: PhantomData,
        }
    }
}

impl<
        BlockT: Block<SignedTransactionT>,
        HaltReasonT: HaltReasonTrait,
        SignedTransactionT: ExecutableTransaction,
    > DebugMineBlockResult<BlockT, HaltReasonT, SignedTransactionT>
{
    /// Whether the block contains a transaction with the given hash.
    pub fn has_transaction(&self, transaction_hash: &B256) -> bool {
        self.block
            .transactions()
            .iter()
            .any(|tx| *tx.transaction_hash() == *transaction_hash)
    }
}
