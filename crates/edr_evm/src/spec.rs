use std::{fmt::Debug, marker::PhantomData};

use edr_eth::{
    beacon::{BEACON_ROOTS_ADDRESS, BEACON_ROOTS_BYTECODE},
    block::{self, BlobGas, PartialHeader},
    eips::eip4844,
    l1::{self, BlockEnv, L1ChainSpec},
    log::{ExecutionLog, FilterLog},
    receipt::MapReceiptLogs,
    result::InvalidTransaction,
    spec::{ChainSpec, EthHeaderConstants},
    transaction::ExecutableTransaction,
    Address, B256, U256,
};
use edr_rpc_eth::{spec::RpcSpec, RpcTypeFrom, TransactionConversionError};
pub use revm::EvmWiring;

use crate::{
    block::transaction::TransactionAndBlock,
    evm::PrimitiveEvmWiring,
    hardfork::{self, Activations},
    receipt::{self, ExecutionReceiptBuilder},
    state::Database,
    transaction::{
        remote::EthRpcTransaction, Transaction, TransactionError, TransactionType,
        TransactionValidation,
    },
    BlockReceipt, EthBlockData, EthRpcBlock, RemoteBlockConversionError,
};

/// A trait for defining a chain's associated types.
// Bug: https://github.com/rust-lang/rust-clippy/issues/12927
#[allow(clippy::trait_duplication_in_bounds)]
pub trait RuntimeSpec:
    alloy_rlp::Encodable
    // Defines the chain's internal types like blocks/headers or transactions
    + EthHeaderConstants
    + ChainSpec<
        Block: BlockEnvConstructor<block::Header> + BlockEnvConstructor<PartialHeader> + Default,
        SignedTransaction: alloy_rlp::Encodable
          + Clone
          + Debug
          + Default
          + PartialEq
          + Eq
          + ExecutableTransaction
          + Transaction
          + TransactionType,
    >
    // Defines an RPC spec and conversion between RPC <-> EVM types
    + RpcSpec<
        RpcBlock<<Self as RpcSpec>::RpcTransaction>: EthRpcBlock
          + TryInto<EthBlockData<Self>, Error = Self::RpcBlockConversionError>,
        RpcReceipt: Debug
          + RpcTypeFrom<BlockReceipt<Self>, Hardfork = Self::Hardfork>
          + TryInto<BlockReceipt<Self>, Error = Self::RpcReceiptConversionError>,
        RpcTransaction: EthRpcTransaction
          + RpcTypeFrom<TransactionAndBlock<Self>, Hardfork = Self::Hardfork>
          + TryInto<Self::SignedTransaction, Error = Self::RpcTransactionConversionError>,
    >
    + RpcSpec<ExecutionReceipt<FilterLog>: Debug>
    + RpcSpec<
        ExecutionReceipt<ExecutionLog>: alloy_rlp::Encodable
          + MapReceiptLogs<ExecutionLog, FilterLog, Self::ExecutionReceipt<FilterLog>>,
        RpcBlock<B256>: EthRpcBlock,
    >
    + Sized
{
    /// Type representing an implementation of `EvmWiring` for this chain.
    type EvmWiring<DatabaseT: Database, ExternalContexT>: EvmWiring<
        ExternalContext = ExternalContexT,
        ChainContext = <Self as ChainSpec>::Context,
        Database = DatabaseT,
        Block = <Self as ChainSpec>::Block,
        Transaction = <Self as ChainSpec>::SignedTransaction,
        Hardfork = <Self as ChainSpec>::Hardfork,
        HaltReason = <Self as ChainSpec>::HaltReason
    >;

    /// Type representing a builder that constructs an execution receipt.
    type ReceiptBuilder: ExecutionReceiptBuilder<
        Self::HaltReason,
        Self::Hardfork,
        Self::SignedTransaction,
        Receipt = Self::ExecutionReceipt<ExecutionLog>,
    >;

    /// Type representing an error that occurs when converting an RPC block.
    type RpcBlockConversionError: std::error::Error;

    /// Type representing an error that occurs when converting an RPC receipt.
    type RpcReceiptConversionError: std::error::Error;

    /// Type representing an error that occurs when converting an RPC
    /// transaction.
    type RpcTransactionConversionError: std::error::Error;

    /// The address and bytecode (string) of a contract that should be deployed
    /// in the genesis state, starting at a particular hardfork.
    ///
    /// Preventing deployment of a legacy contract can be achieved by setting
    /// empty bytecode. As such, hardforks should be ordered from oldest to
    /// newest.
    const PRE_DEPLOYS: &'static [(Self::Hardfork, &'static [(Address, &'static [u8])])];

    /// Casts a transaction validation error into a `TransactionError`.
    ///
    /// This is implemented as an associated function to avoid problems when
    /// implementing type conversions for third-party types.
    fn cast_transaction_error<BlockchainErrorT, StateErrorT>(
        error: <Self::SignedTransaction as TransactionValidation>::ValidationError,
    ) -> TransactionError<Self, BlockchainErrorT, StateErrorT>;

    /// Returns the hardfork activations corresponding to the provided chain ID,
    /// if it is associated with this chain specification.
    fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<Self>>;

    /// Returns the name corresponding to the provided chain ID, if it is
    /// associated with this chain specification.
    fn chain_name(chain_id: u64) -> Option<&'static str>;
}

/// A trait for constructing a (partial) block header into an EVM block.
pub trait BlockEnvConstructor<HeaderT> {
    /// Converts the instance into an EVM block.
    fn new_block_env(header: &HeaderT, hardfork: l1::SpecId) -> Self;
}

impl BlockEnvConstructor<PartialHeader> for BlockEnv {
    fn new_block_env(header: &PartialHeader, hardfork: l1::SpecId) -> Self {
        Self {
            number: U256::from(header.number),
            coinbase: header.beneficiary,
            timestamp: U256::from(header.timestamp),
            difficulty: header.difficulty,
            basefee: header.base_fee.unwrap_or(U256::ZERO),
            gas_limit: U256::from(header.gas_limit),
            prevrandao: if hardfork >= l1::SpecId::MERGE {
                Some(header.mix_hash)
            } else {
                None
            },
            blob_excess_gas_and_price: header
                .blob_gas
                .as_ref()
                .map(|BlobGas { excess_gas, .. }| eip4844::BlobExcessGasAndPrice::new(*excess_gas)),
        }
    }
}

impl BlockEnvConstructor<block::Header> for BlockEnv {
    fn new_block_env(header: &block::Header, hardfork: l1::SpecId) -> Self {
        Self {
            number: U256::from(header.number),
            coinbase: header.beneficiary,
            timestamp: U256::from(header.timestamp),
            difficulty: header.difficulty,
            basefee: header.base_fee_per_gas.unwrap_or(U256::ZERO),
            gas_limit: U256::from(header.gas_limit),
            prevrandao: if hardfork >= l1::SpecId::MERGE {
                Some(header.mix_hash)
            } else {
                None
            },
            blob_excess_gas_and_price: header
                .blob_gas
                .as_ref()
                .map(|BlobGas { excess_gas, .. }| eip4844::BlobExcessGasAndPrice::new(*excess_gas)),
        }
    }
}

/// A supertrait for [`RuntimeSpec`] that is safe to send between threads.
pub trait SyncRuntimeSpec:
    RuntimeSpec<
        ExecutionReceipt<FilterLog>: Send + Sync,
        HaltReason: Send + Sync,
        Hardfork: Send + Sync,
        RpcBlockConversionError: Send + Sync,
        RpcReceiptConversionError: Send + Sync,
        SignedTransaction: TransactionValidation<ValidationError: Send + Sync> + Send + Sync,
    > + Send
    + Sync
    + 'static
{
}

impl<ChainSpecT> SyncRuntimeSpec for ChainSpecT where
    ChainSpecT: RuntimeSpec<
            ExecutionReceipt<FilterLog>: Send + Sync,
            HaltReason: Send + Sync,
            Hardfork: Send + Sync,
            RpcBlockConversionError: Send + Sync,
            RpcReceiptConversionError: Send + Sync,
            SignedTransaction: TransactionValidation<ValidationError: Send + Sync> + Send + Sync,
        > + Send
        + Sync
        + 'static
{
}

/// EVM wiring for L1 chains.
pub struct L1Wiring<ChainSpecT: ChainSpec, DatabaseT: Database, ExternalContextT> {
    _phantom: PhantomData<(ChainSpecT, DatabaseT, ExternalContextT)>,
}

impl<ChainSpecT: ChainSpec, DatabaseT: Database, ExternalContextT> PrimitiveEvmWiring
    for L1Wiring<ChainSpecT, DatabaseT, ExternalContextT>
{
    type ExternalContext = ExternalContextT;
    type ChainContext = ChainSpecT::Context;
    type Database = DatabaseT;
    type Block = ChainSpecT::Block;
    type Transaction = ChainSpecT::SignedTransaction;
    type Hardfork = ChainSpecT::Hardfork;
    type HaltReason = ChainSpecT::HaltReason;
}

impl<ChainSpecT, DatabaseT, ExternalContextT> revm::EvmWiring
    for L1Wiring<ChainSpecT, DatabaseT, ExternalContextT>
where
    ChainSpecT: ChainSpec<
        Block: Default,
        SignedTransaction: Default
                               + TransactionValidation<ValidationError: From<InvalidTransaction>>,
    >,
    DatabaseT: Database,
{
    fn handler<'evm>(hardfork: Self::Hardfork) -> revm::EvmHandler<'evm, Self> {
        revm::EvmHandler::mainnet_with_spec(hardfork)
    }
}

impl RuntimeSpec for L1ChainSpec {
    type EvmWiring<DatabaseT: Database, ExternalContexT> =
        L1Wiring<Self, DatabaseT, ExternalContexT>;

    type ReceiptBuilder = receipt::Builder;
    type RpcBlockConversionError = RemoteBlockConversionError<Self>;
    type RpcReceiptConversionError = edr_rpc_eth::receipt::ConversionError;
    type RpcTransactionConversionError = TransactionConversionError;

    const PRE_DEPLOYS: &'static [(Self::Hardfork, &'static [(Address, &'static [u8])])] = &[(
        l1::SpecId::CANCUN,
        &[(BEACON_ROOTS_ADDRESS, &BEACON_ROOTS_BYTECODE)],
    )];

    fn cast_transaction_error<BlockchainErrorT, StateErrorT>(
        error: <Self::SignedTransaction as TransactionValidation>::ValidationError,
    ) -> TransactionError<Self, BlockchainErrorT, StateErrorT> {
        match error {
            InvalidTransaction::LackOfFundForMaxFee { fee, balance } => {
                TransactionError::LackOfFundForMaxFee { fee, balance }
            }
            remainder => TransactionError::InvalidTransaction(remainder),
        }
    }

    fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<Self>> {
        hardfork::l1::chain_hardfork_activations(chain_id)
    }

    fn chain_name(chain_id: u64) -> Option<&'static str> {
        hardfork::l1::chain_name(chain_id)
    }
}
