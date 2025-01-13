use core::fmt::Debug;

use edr_eth::block::PartialHeader;
use edr_evm::{
    state::{DatabaseComponents, WrapDatabaseRef},
    BlockBuilder, BlockBuilderAndError, EthBlockBuilder, MineBlockResultAndState,
};
use revm_optimism::{L1BlockInfo, OptimismHaltReason};

use crate::{
    block::LocalBlock, receipt::BlockReceiptFactory, transaction, OptimismChainSpec, OptimismSpecId,
};

/// Block builder for Optimism.
pub struct Builder<'blockchain, BlockchainErrorT, DebugDataT, StateErrorT> {
    eth: EthBlockBuilder<'blockchain, BlockchainErrorT, OptimismChainSpec, DebugDataT, StateErrorT>,
    hardfork: OptimismSpecId,
    l1_block_info: L1BlockInfo,
}

impl<'blockchain, BlockchainErrorT, DebugDataT, StateErrorT: Debug + Send>
    BlockBuilder<'blockchain, OptimismChainSpec, DebugDataT>
    for Builder<'blockchain, BlockchainErrorT, DebugDataT, StateErrorT>
{
    type BlockchainError = BlockchainErrorT;
    type StateError = StateErrorT;

    fn new_block_builder(
        blockchain: &'blockchain dyn edr_evm::blockchain::SyncBlockchain<
            OptimismChainSpec,
            Self::BlockchainError,
            Self::StateError,
        >,
        state: Box<dyn edr_evm::state::SyncState<Self::StateError>>,
        hardfork: OptimismSpecId,
        cfg: edr_evm::config::CfgEnv,
        options: edr_eth::block::BlockOptions,
        debug_context: Option<
            edr_evm::DebugContext<
                'blockchain,
                OptimismChainSpec,
                Self::BlockchainError,
                DebugDataT,
                Box<dyn edr_evm::state::SyncState<Self::StateError>>,
            >,
        >,
    ) -> Result<
        Self,
        edr_evm::BlockBuilderCreationError<Self::BlockchainError, OptimismSpecId, Self::StateError>,
    > {
        let mut db = WrapDatabaseRef(DatabaseComponents { blockchain, state });
        let l1_block_info = revm_optimism::L1BlockInfo::try_fetch(&mut db, hardfork)?;
        let DatabaseComponents { blockchain, state } = db.0;

        let eth = EthBlockBuilder::new(blockchain, state, hardfork, cfg, options, debug_context)?;

        Ok(Self {
            eth,
            hardfork,
            l1_block_info,
        })
    }

    fn block_receipt_factory(&self) -> BlockReceiptFactory {
        BlockReceiptFactory {
            l1_block_info: self.l1_block_info.clone(),
        }
    }

    fn header(&self) -> &PartialHeader {
        self.eth.header()
    }

    fn add_transaction(
        self,
        transaction: transaction::Signed,
    ) -> Result<
        Self,
        edr_evm::BlockBuilderAndError<
            Self,
            edr_evm::BlockTransactionError<
                OptimismChainSpec,
                Self::BlockchainError,
                Self::StateError,
            >,
        >,
    > {
        let Self {
            eth,
            hardfork,
            l1_block_info,
        } = self;

        match eth.add_transaction(transaction) {
            Ok(eth) => Ok(Self {
                eth,
                hardfork,
                l1_block_info,
            }),
            Err(BlockBuilderAndError {
                block_builder,
                error,
            }) => Err(BlockBuilderAndError {
                block_builder: Self {
                    eth: block_builder,
                    hardfork,
                    l1_block_info,
                },
                error,
            }),
        }
    }

    fn finalize(
        self,
        rewards: Vec<(edr_eth::Address, edr_eth::U256)>,
    ) -> Result<
        MineBlockResultAndState<OptimismHaltReason, LocalBlock, Self::StateError>,
        Self::StateError,
    > {
        let receipt_factory = self.block_receipt_factory();
        self.eth.finalize(&receipt_factory, rewards)
    }
}
