use edr_eth::{eips::eip2718::TypedEnvelope, l1::L1ChainSpec, receipt::ExecutionReceipt};
use serde::{de::DeserializeOwned, Serialize};

use crate::{receipt::Block, CallRequest};

/// Trait for specifying Ethereum-based JSON-RPC method types.
pub trait RpcSpec {
    /// Type representing an RPC execution receipt.
    type ExecutionReceipt<LogT>: ExecutionReceipt<Log = LogT>;

    /// Type representing an RPC block
    type RpcBlock<DataT>: GetBlockNumber + DeserializeOwned + Serialize
    where
        DataT: Default + DeserializeOwned + Serialize;

    /// Type representing an RPC `eth_call` request.
    type RpcCallRequest: DeserializeOwned + Serialize;

    /// Type representing an RPC receipt.
    type RpcReceipt: DeserializeOwned + Serialize;

    /// Type representing an RPC transaction.
    type RpcTransaction: Default + DeserializeOwned + Serialize;

    /// Type representing an RPC `eth_sendTransaction` request.
    type RpcTransactionRequest: DeserializeOwned + Serialize;
}

pub trait GetBlockNumber {
    fn number(&self) -> Option<u64>;
}

impl RpcSpec for L1ChainSpec {
    type ExecutionReceipt<LogT> = TypedEnvelope<edr_eth::receipt::Execution<LogT>>;
    type RpcBlock<DataT>
        = crate::block::Block<DataT>
    where
        DataT: Default + DeserializeOwned + Serialize;
    type RpcCallRequest = CallRequest;
    type RpcReceipt = Block;
    type RpcTransaction = crate::transaction::TransactionWithSignature;
    type RpcTransactionRequest = crate::transaction::TransactionRequest;
}
