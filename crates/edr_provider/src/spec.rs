use core::fmt::Debug;
use std::sync::Arc;

pub use edr_eth::spec::EthHeaderConstants;
use edr_eth::{
    eips::eip2930,
    l1::L1ChainSpec,
    result::HaltReason,
    rlp,
    transaction::{
        signed::{FakeSign, Sign},
        IsSupported, Transaction,
    },
    Address, Blob, BlockSpec, B256, U256,
};
pub use edr_evm::spec::{RuntimeSpec, SyncRuntimeSpec};
use edr_evm::{
    blockchain::BlockchainErrorForChainSpec, state::StateOverrides, transaction,
    BlockAndTotalDifficulty, BlockReceipts,
};
use edr_rpc_eth::{CallRequest, TransactionRequest};

use crate::{data::ProviderData, time::TimeSinceEpoch, ProviderError, TransactionFailureReason};

pub trait ProviderSpec<TimerT: Clone + TimeSinceEpoch>:
    RuntimeSpec<
    Block: BlockReceipts<Arc<Self::BlockReceipt>, Error = BlockchainErrorForChainSpec<Self>>,
    LocalBlock: BlockReceipts<Arc<Self::BlockReceipt>, Error = BlockchainErrorForChainSpec<Self>>,
    RpcBlock<B256>: From<BlockAndTotalDifficulty<Arc<Self::Block>, Self::SignedTransaction>>,
    RpcCallRequest: MaybeSender,
    RpcTransactionRequest: Sender,
    SignedTransaction: IsSupported,
>
{
    type PooledTransaction: HardforkValidationData
        + Into<Self::SignedTransaction>
        + rlp::Decodable
        + Transaction;

    /// Type representing a transaction request.
    type TransactionRequest: FakeSign<Signed = Self::SignedTransaction>
        + Sign<Signed = Self::SignedTransaction>
        + for<'context> FromRpcType<
            Self::RpcCallRequest,
            TimerT,
            Context<'context> = CallContext<'context, Self, TimerT>,
            Error = ProviderError<Self>,
        > + for<'context> FromRpcType<
            Self::RpcTransactionRequest,
            TimerT,
            Context<'context> = TransactionContext<'context, Self, TimerT>,
            Error = ProviderError<Self>,
        >;

    /// Casts a halt reason into a transaction failure reason.
    ///
    /// This is implemented as an associated function to avoid problems when
    /// implementing type conversions for third-party types.
    fn cast_halt_reason(reason: Self::HaltReason) -> TransactionFailureReason<Self::HaltReason>;
}

impl<TimerT: Clone + TimeSinceEpoch> ProviderSpec<TimerT> for L1ChainSpec {
    type PooledTransaction = transaction::pooled::PooledTransaction;
    type TransactionRequest = transaction::Request;

    fn cast_halt_reason(reason: Self::HaltReason) -> TransactionFailureReason<Self::HaltReason> {
        match reason {
            HaltReason::CreateContractSizeLimit => {
                TransactionFailureReason::CreateContractSizeLimit
            }
            HaltReason::OpcodeNotFound | HaltReason::InvalidFEOpcode => {
                TransactionFailureReason::OpcodeNotFound
            }
            HaltReason::OutOfGas(error) => TransactionFailureReason::OutOfGas(error),
            remainder => TransactionFailureReason::Inner(remainder),
        }
    }
}

/// Trait with data used for validating a transaction complies with a
/// [`edr_eth::l1::SpecId`].
pub trait HardforkValidationData {
    /// Returns the `to` address of the transaction.
    fn to(&self) -> Option<&Address>;

    /// Returns the gas price of the transaction.
    fn gas_price(&self) -> Option<&U256>;

    /// Returns the max fee per gas of the transaction.
    fn max_fee_per_gas(&self) -> Option<&U256>;

    /// Returns the max priority fee per gas of the transaction.
    fn max_priority_fee_per_gas(&self) -> Option<&U256>;

    /// Returns the access list of the transaction.
    fn access_list(&self) -> Option<&Vec<eip2930::AccessListItem>>;

    /// Returns the blobs of the transaction.
    fn blobs(&self) -> Option<&Vec<Blob>>;

    /// Returns the blob hashes of the transaction.
    fn blob_hashes(&self) -> Option<&Vec<B256>>;
}

/// Trait for retrieving the sender of a request, if any.
pub trait MaybeSender {
    /// Retrieves the sender of the request, if any.
    fn maybe_sender(&self) -> Option<&Address>;
}

impl MaybeSender for CallRequest {
    fn maybe_sender(&self) -> Option<&Address> {
        self.from.as_ref()
    }
}

/// Trait for retrieving the sender of a request.
pub trait Sender {
    /// Retrieves the sender of the request.
    fn sender(&self) -> &Address;
}

impl Sender for TransactionRequest {
    fn sender(&self) -> &Address {
        &self.from
    }
}

// ChainSpecT: ProviderSpec<TimerT, Hardfork: Debug>,

/// Trait for resolving an RPC type to an internal type.
pub trait FromRpcType<RpcT, TimerT: Clone + TimeSinceEpoch>: Sized {
    /// Type for contextual information.
    type Context<'context>;

    /// Type of error that can occur during resolution.
    type Error;

    fn from_rpc_type(value: RpcT, context: Self::Context<'_>) -> Result<Self, Self::Error>;
}

pub trait SyncProviderSpec<TimerT: Clone + TimeSinceEpoch>:
    ProviderSpec<TimerT> + SyncRuntimeSpec
{
}

impl<ProviderSpecT: ProviderSpec<TimerT> + SyncRuntimeSpec, TimerT: Clone + TimeSinceEpoch>
    SyncProviderSpec<TimerT> for ProviderSpecT
{
}

pub type DefaultGasPriceFn<ChainSpecT, TimerT> =
    fn(&ProviderData<ChainSpecT, TimerT>) -> Result<U256, ProviderError<ChainSpecT>>;

pub type MaxFeesFn<ChainSpecT, TimerT> = fn(
    &ProviderData<ChainSpecT, TimerT>,
    // block_spec
    &BlockSpec,
    // max_fee_per_gas
    Option<U256>,
    // max_priority_fee_per_gas
    Option<U256>,
) -> Result<(U256, U256), ProviderError<ChainSpecT>>;

pub struct CallContext<
    'context,
    ChainSpecT: ProviderSpec<TimerT, Hardfork: Debug>,
    TimerT: Clone + TimeSinceEpoch,
> {
    pub data: &'context mut ProviderData<ChainSpecT, TimerT>,
    pub block_spec: &'context BlockSpec,
    pub state_overrides: &'context StateOverrides,
    pub default_gas_price_fn: DefaultGasPriceFn<ChainSpecT, TimerT>,
    pub max_fees_fn: MaxFeesFn<ChainSpecT, TimerT>,
}

pub struct TransactionContext<
    'context,
    ChainSpecT: ProviderSpec<TimerT, Hardfork: Debug>,
    TimerT: Clone + TimeSinceEpoch,
> {
    pub data: &'context mut ProviderData<ChainSpecT, TimerT>,
}
