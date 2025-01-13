use std::ops::Deref;

use alloy_rlp::BufMut;

use super::{AsExecutionReceipt, ExecutionReceipt, ReceiptTrait, RootOrStatus, TransactionReceipt};
use crate::{log::FilterLog, Address, Bloom, B256, U256};

/// Type for a receipt that's included in a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockReceipt<ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>> {
    pub inner: TransactionReceipt<ExecutionReceiptT>,
    /// Hash of the block that this is part of
    pub block_hash: B256,
    /// Number of the block that this is part of
    pub block_number: u64,
}

impl<ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>> AsExecutionReceipt
    for BlockReceipt<ExecutionReceiptT>
{
    type ExecutionReceipt = ExecutionReceiptT;

    fn as_execution_receipt(&self) -> &ExecutionReceiptT {
        self.inner.as_execution_receipt()
    }
}

impl<ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>> Deref
    for BlockReceipt<ExecutionReceiptT>
{
    type Target = TransactionReceipt<ExecutionReceiptT>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<ExecutionReceiptT> alloy_rlp::Encodable for BlockReceipt<ExecutionReceiptT>
where
    ExecutionReceiptT: ExecutionReceipt<Log = FilterLog> + alloy_rlp::Encodable,
{
    fn encode(&self, out: &mut dyn BufMut) {
        self.inner.encode(out);
    }

    fn length(&self) -> usize {
        self.inner.length()
    }
}

impl<ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>> ExecutionReceipt
    for BlockReceipt<ExecutionReceiptT>
{
    type Log = ExecutionReceiptT::Log;

    fn cumulative_gas_used(&self) -> u64 {
        self.inner.cumulative_gas_used()
    }

    fn logs_bloom(&self) -> &Bloom {
        self.inner.logs_bloom()
    }

    fn transaction_logs(&self) -> &[Self::Log] {
        self.inner.transaction_logs()
    }

    fn root_or_status(&self) -> RootOrStatus<'_> {
        self.inner.root_or_status()
    }
}

impl<ExecutionReceiptT> ReceiptTrait for BlockReceipt<ExecutionReceiptT>
where
    ExecutionReceiptT: ExecutionReceipt<Log = FilterLog>,
{
    fn block_number(&self) -> u64 {
        self.block_number
    }

    fn block_hash(&self) -> &B256 {
        &self.block_hash
    }

    fn contract_address(&self) -> Option<&Address> {
        self.inner.contract_address.as_ref()
    }

    fn effective_gas_price(&self) -> Option<&U256> {
        self.inner.effective_gas_price.as_ref()
    }

    fn from(&self) -> &Address {
        &self.inner.from
    }

    fn gas_used(&self) -> u64 {
        self.inner.gas_used
    }

    fn to(&self) -> Option<&Address> {
        self.inner.to.as_ref()
    }

    fn transaction_hash(&self) -> &B256 {
        &self.inner.transaction_hash
    }

    fn transaction_index(&self) -> u64 {
        self.inner.transaction_index
    }
}
