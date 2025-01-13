use edr_eth::{
    log::FilterLog,
    receipt::{AsExecutionReceipt, ExecutionReceipt, ReceiptTrait, RootOrStatus},
    Address, Bloom, B256, U256,
};
use op_alloy_rpc_types::receipt::L1BlockInfo;

use crate::{eip2718::TypedEnvelope, receipt};

/// An Optimism block receipt.
///
/// Includes the L1 block info for non-deposit transactions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The underlying Ethereum block receipt.
    pub eth: edr_eth::receipt::BlockReceipt<TypedEnvelope<receipt::Execution<FilterLog>>>,
    /// The L1 block info, if not a deposit transaction.
    pub l1_block_info: Option<L1BlockInfo>,
}

impl AsExecutionReceipt for Block {
    type ExecutionReceipt = TypedEnvelope<receipt::Execution<FilterLog>>;

    fn as_execution_receipt(&self) -> &Self::ExecutionReceipt {
        self.eth.as_execution_receipt()
    }
}

impl alloy_rlp::Encodable for Block {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        self.eth.encode(out);
    }

    fn length(&self) -> usize {
        self.eth.length()
    }
}

impl ExecutionReceipt for Block {
    type Log = FilterLog;

    fn cumulative_gas_used(&self) -> u64 {
        self.eth.cumulative_gas_used()
    }

    fn logs_bloom(&self) -> &Bloom {
        self.eth.logs_bloom()
    }

    fn transaction_logs(&self) -> &[Self::Log] {
        self.eth.transaction_logs()
    }

    fn root_or_status(&self) -> RootOrStatus<'_> {
        self.eth.root_or_status()
    }
}

impl ReceiptTrait for Block {
    fn block_number(&self) -> u64 {
        self.eth.block_number()
    }

    fn block_hash(&self) -> &B256 {
        self.eth.block_hash()
    }

    fn contract_address(&self) -> Option<&Address> {
        self.eth.contract_address()
    }

    fn effective_gas_price(&self) -> Option<&U256> {
        self.eth.effective_gas_price()
    }

    fn from(&self) -> &Address {
        self.eth.from()
    }

    fn gas_used(&self) -> u64 {
        self.eth.gas_used()
    }

    fn to(&self) -> Option<&Address> {
        self.eth.to()
    }

    fn transaction_hash(&self) -> &B256 {
        self.eth.transaction_hash()
    }

    fn transaction_index(&self) -> u64 {
        self.eth.transaction_index()
    }
}
