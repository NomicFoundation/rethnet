use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc};

use alloy_rlp::RlpEncodable;
use edr_eth::{
    eips::eip1559::{BaseFeeParams, ConstantBaseFeeParams, ForkBaseFeeParams},
    l1,
    result::{HaltReason, InvalidTransaction},
    spec::{ChainSpec, EthHeaderConstants},
};
use edr_evm::{
    evm::{
        handler::register::{EvmHandler, HandleRegisters},
        EvmWiring, PrimitiveEvmWiring,
    },
    spec::RuntimeSpec,
    state::Database,
    transaction::{TransactionError, TransactionValidation},
    BlockReceipts, RemoteBlock, RemoteBlockConversionError, SyncBlock,
};
use edr_napi_core::{
    napi,
    spec::{marshal_response_data, Response, SyncNapiSpec},
};
use edr_provider::{time::TimeSinceEpoch, ProviderSpec, TransactionFailureReason};
use edr_rpc_eth::{jsonrpc, spec::RpcSpec};
use edr_solidity::contract_decoder::ContractDecoder;
use revm_optimism::{OptimismHaltReason, OptimismInvalidTransaction, OptimismSpecId};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    block::{self, LocalBlock},
    eip2718::TypedEnvelope,
    hardfork,
    receipt::{self, BlockReceiptFactory},
    rpc, transaction,
};

/// Chain specification for the Ethereum JSON-RPC API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, RlpEncodable)]
pub struct OptimismChainSpec;

impl RpcSpec for OptimismChainSpec {
    type ExecutionReceipt<Log> = TypedEnvelope<receipt::Execution<Log>>;
    type RpcBlock<Data>
        = edr_rpc_eth::Block<Data>
    where
        Data: Default + DeserializeOwned + Serialize;
    type RpcCallRequest = edr_rpc_eth::CallRequest;
    type RpcReceipt = rpc::BlockReceipt;
    type RpcTransaction = rpc::Transaction;
    type RpcTransactionRequest = edr_rpc_eth::TransactionRequest;
}

impl ChainSpec for OptimismChainSpec {
    type BlockEnv = l1::BlockEnv;
    type Context = revm_optimism::Context;
    type HaltReason = OptimismHaltReason;
    type Hardfork = OptimismSpecId;
    type SignedTransaction = transaction::Signed;
}

/// EVM wiring for Optimism chains.
pub struct Wiring<DatabaseT: Database, ExternalContextT> {
    _phantom: PhantomData<(DatabaseT, ExternalContextT)>,
}

impl<DatabaseT, ExternalContextT> PrimitiveEvmWiring for Wiring<DatabaseT, ExternalContextT>
where
    DatabaseT: Database,
{
    type ExternalContext = ExternalContextT;
    type ChainContext = <OptimismChainSpec as ChainSpec>::Context;
    type Database = DatabaseT;
    type Block = <OptimismChainSpec as ChainSpec>::BlockEnv;
    type Transaction = <OptimismChainSpec as ChainSpec>::SignedTransaction;
    type Hardfork = <OptimismChainSpec as ChainSpec>::Hardfork;
    type HaltReason = <OptimismChainSpec as ChainSpec>::HaltReason;
}

impl<DatabaseT, ExternalContextT> EvmWiring for Wiring<DatabaseT, ExternalContextT>
where
    DatabaseT: Database,
{
    fn handler<'evm>(hardfork: Self::Hardfork) -> EvmHandler<'evm, Self> {
        let mut handler = EvmHandler::mainnet_with_spec(hardfork);

        handler.append_handler_register(HandleRegisters::Plain(
            revm_optimism::optimism_handle_register::<Wiring<DatabaseT, ExternalContextT>>,
        ));

        handler
    }
}

impl RuntimeSpec for OptimismChainSpec {
    type Block = dyn SyncBlock<
        Arc<Self::BlockReceipt>,
        Self::SignedTransaction,
        Error = <Self::LocalBlock as BlockReceipts<Arc<Self::BlockReceipt>>>::Error,
    >;

    type BlockBuilder<
        'blockchain,
        BlockchainErrorT: 'blockchain,
        DebugDataT,
        StateErrorT: 'blockchain + Debug + Send,
    > = block::Builder<'blockchain, BlockchainErrorT, DebugDataT, StateErrorT>;

    type BlockReceipt = receipt::Block;
    type BlockReceiptFactory = BlockReceiptFactory;

    type EvmWiring<DatabaseT: Database, ExternalContexT> = Wiring<DatabaseT, ExternalContexT>;
    type LocalBlock = LocalBlock;

    type ReceiptBuilder = receipt::execution::Builder;
    type RpcBlockConversionError = RemoteBlockConversionError<Self::RpcTransactionConversionError>;
    type RpcReceiptConversionError = rpc::receipt::ConversionError;
    type RpcTransactionConversionError = rpc::transaction::ConversionError;

    fn cast_local_block(local_block: Arc<Self::LocalBlock>) -> Arc<Self::Block> {
        local_block
    }

    fn cast_remote_block(remote_block: Arc<RemoteBlock<Self>>) -> Arc<Self::Block> {
        remote_block
    }

    fn cast_transaction_error<BlockchainErrorT, StateErrorT>(
        error: <Self::SignedTransaction as TransactionValidation>::ValidationError,
    ) -> TransactionError<Self, BlockchainErrorT, StateErrorT> {
        match error {
            OptimismInvalidTransaction::Base(InvalidTransaction::LackOfFundForMaxFee {
                fee,
                balance,
            }) => TransactionError::LackOfFundForMaxFee { fee, balance },
            remainder => TransactionError::InvalidTransaction(remainder),
        }
    }

    fn chain_hardfork_activations(
        chain_id: u64,
    ) -> Option<&'static edr_evm::hardfork::Activations<Self::Hardfork>> {
        hardfork::chain_hardfork_activations(chain_id)
    }

    fn chain_name(chain_id: u64) -> Option<&'static str> {
        hardfork::chain_name(chain_id)
    }
}

impl EthHeaderConstants for OptimismChainSpec {
    const BASE_FEE_PARAMS: BaseFeeParams<OptimismSpecId> =
        BaseFeeParams::Variable(ForkBaseFeeParams::new(&[
            (OptimismSpecId::LONDON, ConstantBaseFeeParams::new(50, 6)),
            (OptimismSpecId::CANYON, ConstantBaseFeeParams::new(250, 6)),
        ]));

    const MIN_ETHASH_DIFFICULTY: u64 = 0;
}

impl SyncNapiSpec for OptimismChainSpec {
    const CHAIN_TYPE: &'static str = "Optimism";

    fn cast_response(
        response: Result<
            edr_provider::ResponseWithTraces<OptimismHaltReason>,
            edr_provider::ProviderError<Self>,
        >,
        _contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<edr_napi_core::spec::Response<HaltReason>> {
        let response = jsonrpc::ResponseData::from(response.map(|response| response.result));

        marshal_response_data(response).map(|data| Response {
            data,
            // TODO: Add support for Solidity stack traces in Optimism
            solidity_trace: None,
            traces: Vec::new(),
        })
    }
}

impl<TimerT: Clone + TimeSinceEpoch> ProviderSpec<TimerT> for OptimismChainSpec {
    type PooledTransaction = transaction::Pooled;
    type TransactionRequest = transaction::Request;

    fn cast_halt_reason(
        reason: OptimismHaltReason,
    ) -> TransactionFailureReason<OptimismHaltReason> {
        match reason {
            OptimismHaltReason::Base(reason) => match reason {
                HaltReason::CreateContractSizeLimit => {
                    TransactionFailureReason::CreateContractSizeLimit
                }
                HaltReason::OpcodeNotFound | HaltReason::InvalidFEOpcode => {
                    TransactionFailureReason::OpcodeNotFound
                }
                HaltReason::OutOfGas(error) => TransactionFailureReason::OutOfGas(error),
                remainder => TransactionFailureReason::Inner(OptimismHaltReason::Base(remainder)),
            },
            remainder => TransactionFailureReason::Inner(remainder),
        }
    }
}
