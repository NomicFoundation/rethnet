mod l1;

use std::fmt::Debug;

use edr_eth::{
    block::{BlockOptions, PartialHeader},
    spec::ChainSpec,
    Address, U256,
};

pub use self::l1::{EthBlockBuilder, EthBlockReceiptFactory};
use crate::{
    blockchain::SyncBlockchain,
    config::CfgEnv,
    debug::DebugContextForChainSpec,
    spec::RuntimeSpec,
    state::{DatabaseComponentError, SyncState},
    transaction::TransactionError,
    MineBlockResultAndStateForChainSpec,
};

/// An error caused during construction of a block builder.
#[derive(Debug, thiserror::Error)]
pub enum BlockBuilderCreationError<BlockchainErrorT, HardforkT, StateErrorT> {
    /// Blockchain error
    #[error(transparent)]
    Blockchain(BlockchainErrorT),
    /// State error
    #[error(transparent)]
    State(StateErrorT),
    /// Unsupported hardfork. Hardforks older than Byzantium are not supported
    #[error("Unsupported hardfork: {0:?}. Hardforks older than Byzantium are not supported.")]
    UnsupportedHardfork(HardforkT),
}

/// Helper type for a chain-specific [`BlockBuilderCreationError`].
pub type BlockBuilderCreationErrorForChainSpec<BlockchainErrorT, ChainSpecT, StateErrorT> =
    BlockBuilderCreationError<BlockchainErrorT, <ChainSpecT as ChainSpec>::Hardfork, StateErrorT>;

impl<BlockchainErrorT, HardforkT: Debug, StateErrorT>
    From<DatabaseComponentError<BlockchainErrorT, StateErrorT>>
    for BlockBuilderCreationError<BlockchainErrorT, HardforkT, StateErrorT>
{
    fn from(value: DatabaseComponentError<BlockchainErrorT, StateErrorT>) -> Self {
        match value {
            DatabaseComponentError::Blockchain(error) => Self::Blockchain(error),
            DatabaseComponentError::State(error) => Self::State(error),
        }
    }
}

/// An error caused during execution of a transaction while building a block.
#[derive(Debug, thiserror::Error)]
pub enum BlockTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: ChainSpec,
{
    /// Transaction has higher gas limit than is remaining in block
    #[error("Transaction has a higher gas limit than the remaining gas in the block")]
    ExceedsBlockGasLimit,
    /// Transaction has higher blob gas usage than is remaining in block
    #[error("Transaction has higher blob gas usage than is remaining in block")]
    ExceedsBlockBlobGasLimit,
    /// Transaction error
    #[error(transparent)]
    Transaction(#[from] TransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>),
}

/// Helper type for a [`BlockBuilderAndError`] with a [`BlockTransactionError`].
pub type BlockBuilderAndTransactionError<BlockBuilderT, BlockchainErrorT, ChainSpecT, StateErrorT> =
    BlockBuilderAndError<
        BlockBuilderT,
        BlockTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>,
    >;

/// A wrapper around a block builder and an error.
pub struct BlockBuilderAndError<BlockBuilderT, ErrorT> {
    /// The block builder.
    pub block_builder: BlockBuilderT,
    /// The error.
    pub error: ErrorT,
}

/// A trait for building blocks.
pub trait BlockBuilder<'blockchain, ChainSpecT, DebugDataT>: Sized
where
    ChainSpecT: RuntimeSpec,
{
    /// The blockchain's error type.
    type BlockchainError;

    /// The state's error type.
    type StateError: Debug + Send;

    /// Creates a new block builder.
    fn new_block_builder(
        blockchain: &'blockchain dyn SyncBlockchain<
            ChainSpecT,
            Self::BlockchainError,
            Self::StateError,
        >,
        state: Box<dyn SyncState<Self::StateError>>,
        hardfork: ChainSpecT::Hardfork,
        cfg: CfgEnv,
        options: BlockOptions,
        debug_context: Option<
            DebugContextForChainSpec<
                'blockchain,
                Self::BlockchainError,
                ChainSpecT,
                DebugDataT,
                Self::StateError,
            >,
        >,
    ) -> Result<
        Self,
        BlockBuilderCreationErrorForChainSpec<Self::BlockchainError, ChainSpecT, Self::StateError>,
    >;

    /// Returns the block's receipt factory.
    fn block_receipt_factory(&self) -> ChainSpecT::BlockReceiptFactory;

    /// Returns the block's [`PartialHeader`].
    fn header(&self) -> &PartialHeader;

    /// Adds a transaction to the block.
    fn add_transaction(
        self,
        transaction: ChainSpecT::SignedTransaction,
    ) -> Result<
        Self,
        BlockBuilderAndTransactionError<Self, Self::BlockchainError, ChainSpecT, Self::StateError>,
    >;

    /// Finalizes the block, applying rewards to the state.
    fn finalize(
        self,
        rewards: Vec<(Address, U256)>,
    ) -> Result<MineBlockResultAndStateForChainSpec<ChainSpecT, Self::StateError>, Self::StateError>;
}
