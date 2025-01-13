/// Wrapper struct for a transaction and its receipt.
pub struct DetailedTransaction<'transaction, SignedTransactionT, TransactionReceipT> {
    /// The transaction
    pub transaction: &'transaction SignedTransactionT,
    /// The transaction's receipt
    pub receipt: &'transaction TransactionReceipT,
}
