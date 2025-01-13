use edr_rpc_eth::RpcSpec;
use serde::{de::DeserializeOwned, Serialize};

use crate::{eip2718::TypedEnvelope, GenericChainSpec};

pub mod block;
pub mod receipt;
pub mod transaction;

impl RpcSpec for GenericChainSpec {
    type ExecutionReceipt<Log> = TypedEnvelope<edr_eth::receipt::Execution<Log>>;
    type RpcBlock<Data>
        = self::block::Block<Data>
    where
        Data: Default + DeserializeOwned + Serialize;
    type RpcCallRequest = edr_rpc_eth::CallRequest;
    type RpcReceipt = self::receipt::BlockReceipt;
    type RpcTransaction = self::transaction::TransactionWithSignature;
    type RpcTransactionRequest = edr_rpc_eth::TransactionRequest;
}
