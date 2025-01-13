use edr_eth::B256;

/// Trait for retrieving information from an Ethereum JSON-RPC transaction.
pub trait EthRpcTransaction {
    /// Returns the hash of the finalised block associated with the transaction.
    /// If the transaction is pending, returns `None`.
    fn block_hash(&self) -> Option<&B256>;
}

impl EthRpcTransaction for edr_rpc_eth::TransactionWithSignature {
    fn block_hash(&self) -> Option<&B256> {
        self.block_hash.as_ref()
    }
}
