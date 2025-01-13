use std::sync::Arc;

use edr_eth::{
    eips::eip1559::BaseFeeParams,
    l1::{self, L1ChainSpec},
    log::FilterLog,
    receipt::BlockReceipt,
    result::{HaltReason, InvalidTransaction},
    spec::{ChainSpec, EthHeaderConstants},
    transaction::TransactionValidation,
};
use edr_evm::{
    hardfork::Activations,
    spec::{ExecutionReceiptTypeConstructorForChainSpec, L1Wiring, RuntimeSpec},
    state::Database,
    transaction::TransactionError,
    BlockReceipts, EthBlockBuilder, EthBlockReceiptFactory, EthLocalBlock, RemoteBlock, SyncBlock,
};
use edr_provider::{time::TimeSinceEpoch, ProviderSpec, TransactionFailureReason};

use crate::GenericChainSpec;

impl ChainSpec for GenericChainSpec {
    type BlockEnv = l1::BlockEnv;
    type Context = ();
    type HaltReason = HaltReason;
    type Hardfork = l1::SpecId;
    type SignedTransaction = crate::transaction::SignedWithFallbackToPostEip155;
}

impl EthHeaderConstants for GenericChainSpec {
    const BASE_FEE_PARAMS: BaseFeeParams<Self::Hardfork> = L1ChainSpec::BASE_FEE_PARAMS;

    const MIN_ETHASH_DIFFICULTY: u64 = L1ChainSpec::MIN_ETHASH_DIFFICULTY;
}

impl RuntimeSpec for GenericChainSpec {
    type Block = dyn SyncBlock<
        Arc<Self::BlockReceipt>,
        Self::SignedTransaction,
        Error = <Self::LocalBlock as BlockReceipts<Arc<Self::BlockReceipt>>>::Error,
    >;

    type BlockBuilder<
        'blockchain,
        BlockchainErrorT: 'blockchain,
        DebugDataT,
        StateErrorT: 'blockchain + std::fmt::Debug + Send,
    > = EthBlockBuilder<'blockchain, BlockchainErrorT, Self, DebugDataT, StateErrorT>;

    type BlockReceipt = BlockReceipt<Self::ExecutionReceipt<FilterLog>>;

    type BlockReceiptFactory = EthBlockReceiptFactory<Self::ExecutionReceipt<FilterLog>>;

    type EvmWiring<DatabaseT: Database, ExternalContexT> =
        L1Wiring<Self, DatabaseT, ExternalContexT>;

    type LocalBlock = EthLocalBlock<
        Self::RpcBlockConversionError,
        Self::BlockReceipt,
        ExecutionReceiptTypeConstructorForChainSpec<Self>,
        Self::Hardfork,
        Self::RpcReceiptConversionError,
        Self::SignedTransaction,
    >;

    type ReceiptBuilder = crate::receipt::execution::Builder;
    type RpcBlockConversionError = crate::rpc::block::ConversionError<Self>;
    type RpcReceiptConversionError = crate::rpc::receipt::ConversionError;
    type RpcTransactionConversionError = crate::rpc::transaction::ConversionError;

    fn cast_local_block(local_block: Arc<Self::LocalBlock>) -> Arc<Self::Block> {
        local_block
    }

    fn cast_remote_block(remote_block: Arc<RemoteBlock<Self>>) -> Arc<Self::Block> {
        remote_block
    }

    fn cast_transaction_error<BlockchainErrorT, StateErrorT>(
        error: <Self::SignedTransaction as TransactionValidation>::ValidationError,
    ) -> TransactionError<Self, BlockchainErrorT, StateErrorT> {
        // Can't use L1ChainSpec impl here as the TransactionError is generic
        // over the specific chain spec rather than just the validation error.
        // Instead, we copy the impl here.
        match error {
            InvalidTransaction::LackOfFundForMaxFee { fee, balance } => {
                TransactionError::LackOfFundForMaxFee { fee, balance }
            }
            remainder => TransactionError::InvalidTransaction(remainder),
        }
    }

    fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<Self::Hardfork>> {
        L1ChainSpec::chain_hardfork_activations(chain_id)
    }

    fn chain_name(chain_id: u64) -> Option<&'static str> {
        L1ChainSpec::chain_name(chain_id)
    }
}

impl<TimerT: Clone + TimeSinceEpoch> ProviderSpec<TimerT> for GenericChainSpec {
    type PooledTransaction = edr_eth::transaction::pooled::PooledTransaction;
    type TransactionRequest = crate::transaction::Request;

    fn cast_halt_reason(reason: Self::HaltReason) -> TransactionFailureReason<Self::HaltReason> {
        <L1ChainSpec as ProviderSpec<TimerT>>::cast_halt_reason(reason)
    }
}
