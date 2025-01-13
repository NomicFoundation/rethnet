use auto_impl::auto_impl;
use revm_wiring::evm_wiring::HardforkTrait;

use crate::{
    log::FilterLog,
    receipt::{ExecutionReceipt, ReceiptTrait, TransactionReceipt},
    B256,
};

/// Trait for constructing a receipt from a transaction receipt and the block it
/// was executed in.
#[auto_impl(&, Box, Arc)]
pub trait ReceiptFactory<ExecutionReceiptT, HardforkT, SignedTransactionT>
where
    ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>,
    HardforkT: HardforkTrait,
{
    /// Type of the receipt that the factory constructs.
    type Output: ExecutionReceipt<Log = FilterLog> + ReceiptTrait;

    /// Constructs a new instance from a transaction receipt and the block it
    /// was executed in.
    fn create_receipt(
        &self,
        hardfork: HardforkT,
        transaction: &SignedTransactionT,
        transaction_receipt: TransactionReceipt<ExecutionReceiptT>,
        block_hash: &B256,
        block_number: u64,
    ) -> Self::Output;
}
