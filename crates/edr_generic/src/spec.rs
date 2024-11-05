use edr_eth::{
    eips::eip1559::BaseFeeParams,
    l1::{self, L1ChainSpec},
    result::{HaltReason, InvalidTransaction},
    spec::{ChainSpec, EthHeaderConstants},
    transaction::TransactionValidation,
};
use edr_evm::{
    hardfork::Activations,
    spec::{L1Wiring, RuntimeSpec},
    state::Database,
    transaction::TransactionError,
};
use edr_provider::{time::TimeSinceEpoch, ProviderSpec, TransactionFailureReason};

use crate::GenericChainSpec;

impl ChainSpec for GenericChainSpec {
    type Block = l1::BlockEnv;
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
    type EvmWiring<DatabaseT: Database, ExternalContexT> =
        L1Wiring<Self, DatabaseT, ExternalContexT>;

    type ReceiptBuilder = crate::receipt::execution::Builder;
    type RpcBlockConversionError = crate::rpc::block::ConversionError<Self>;
    type RpcReceiptConversionError = crate::rpc::receipt::ConversionError;
    type RpcTransactionConversionError = crate::rpc::transaction::ConversionError;

    const PRE_DEPLOYS: &'static [(Self::Hardfork, &'static [(edr_eth::Address, &'static [u8])])] =
        <L1ChainSpec as RuntimeSpec>::PRE_DEPLOYS;

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

    fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<Self>> {
        L1ChainSpec::chain_hardfork_activations(chain_id).map(Activations::as_chain_spec)
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
