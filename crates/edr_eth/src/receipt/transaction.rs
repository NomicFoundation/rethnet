use alloy_rlp::BufMut;

use super::{AsExecutionReceipt, ExecutionReceipt, MapReceiptLogs};
use crate::{
    l1,
    result::{ExecutionResult, Output},
    spec::{HaltReasonTrait, HardforkTrait},
    transaction::{ExecutableTransaction, Transaction, TransactionType},
    Address, Bloom, B256, U256,
};

/// Type for a receipt that's created when processing a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionReceipt<ExecutionReceiptT: ExecutionReceipt> {
    pub inner: ExecutionReceiptT,
    /// Hash of the transaction
    pub transaction_hash: B256,
    /// Index of the transaction in the block
    pub transaction_index: u64,
    /// Address of the sender
    pub from: Address,
    /// Address of the receiver. `None` when it's a contract creation
    /// transaction.
    pub to: Option<Address>,
    /// The contract address created, if the transaction was a contract
    /// creation, otherwise `None`.
    pub contract_address: Option<Address>,
    /// Gas used by this transaction alone.
    pub gas_used: u64,
    /// The actual value per gas deducted from the senders account, which is
    /// equal to baseFeePerGas + min(maxFeePerGas - baseFeePerGas,
    /// maxPriorityFeePerGas) after EIP-1559. Following Hardhat, only present if
    /// the hardfork is at least London.
    pub effective_gas_price: Option<U256>,
}

impl<ExecutionReceiptT: ExecutionReceipt> AsExecutionReceipt
    for TransactionReceipt<ExecutionReceiptT>
{
    type ExecutionReceipt = ExecutionReceiptT;

    fn as_execution_receipt(&self) -> &ExecutionReceiptT {
        &self.inner
    }
}

impl<ExecutionReceiptT: ExecutionReceipt> TransactionReceipt<ExecutionReceiptT> {
    /// Converts the instance into the inner execution receipt.
    pub fn into_execution_receipt(self) -> ExecutionReceiptT {
        self.inner
    }
}

impl<ExecutionReceiptT: ExecutionReceipt> TransactionReceipt<ExecutionReceiptT> {
    /// Constructs a new instance using the provided execution receipt an
    /// transaction
    pub fn new<HaltReasonT: HaltReasonTrait, HardforkT: HardforkTrait>(
        execution_receipt: ExecutionReceiptT,
        transaction: &(impl Transaction + ExecutableTransaction),
        result: &ExecutionResult<HaltReasonT>,
        transaction_index: u64,
        block_base_fee: U256,
        hardfork: HardforkT,
    ) -> Self {
        let contract_address = if let ExecutionResult::Success {
            output: Output::Create(_, address),
            ..
        } = result
        {
            *address
        } else {
            None
        };

        let effective_gas_price = if hardfork.into() >= l1::SpecId::LONDON {
            Some(
                transaction
                    .effective_gas_price(block_base_fee)
                    .unwrap_or_else(|| *transaction.gas_price()),
            )
        } else {
            None
        };

        Self {
            inner: execution_receipt,
            transaction_hash: *transaction.transaction_hash(),
            transaction_index,
            from: *transaction.caller(),
            to: transaction.kind().to().copied(),
            contract_address,
            gas_used: result.gas_used(),
            effective_gas_price,
        }
    }
}

impl<OldExecutionReceiptT, NewExecutionReceiptT, OldLogT, NewLogT>
    MapReceiptLogs<OldLogT, NewLogT, TransactionReceipt<NewExecutionReceiptT>>
    for TransactionReceipt<OldExecutionReceiptT>
where
    OldExecutionReceiptT:
        MapReceiptLogs<OldLogT, NewLogT, NewExecutionReceiptT> + ExecutionReceipt<Log = OldLogT>,
    NewExecutionReceiptT: ExecutionReceipt<Log = NewLogT>,
{
    fn map_logs(
        self,
        map_fn: impl FnMut(OldLogT) -> NewLogT,
    ) -> TransactionReceipt<NewExecutionReceiptT> {
        TransactionReceipt {
            inner: self.inner.map_logs(map_fn),
            transaction_hash: self.transaction_hash,
            transaction_index: self.transaction_index,
            from: self.from,
            to: self.to,
            contract_address: self.contract_address,
            gas_used: self.gas_used,
            effective_gas_price: self.effective_gas_price,
        }
    }
}

impl<ExecutionReceiptT: ExecutionReceipt> ExecutionReceipt
    for TransactionReceipt<ExecutionReceiptT>
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

    fn root_or_status(&self) -> super::RootOrStatus<'_> {
        self.inner.root_or_status()
    }
}

impl<ExecutionReceiptT: ExecutionReceipt + TransactionType> TransactionType
    for TransactionReceipt<ExecutionReceiptT>
{
    type Type = ExecutionReceiptT::Type;

    fn transaction_type(&self) -> Self::Type {
        self.inner.transaction_type()
    }
}

impl<ExecutionReceiptT> alloy_rlp::Encodable for TransactionReceipt<ExecutionReceiptT>
where
    ExecutionReceiptT: ExecutionReceipt + alloy_rlp::Encodable,
{
    fn encode(&self, out: &mut dyn BufMut) {
        self.inner.encode(out);
    }

    fn length(&self) -> usize {
        self.inner.length()
    }
}
