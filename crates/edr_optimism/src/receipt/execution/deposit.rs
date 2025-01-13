use alloy_rlp::{RlpDecodable, RlpEncodable};
use edr_eth::{receipt::MapReceiptLogs, Bloom};

use super::{Deposit, Execution};

#[derive(RlpDecodable)]
#[rlp(trailing)]
pub(super) struct Eip658OrDeposit<LogT> {
    status: bool,
    cumulative_gas_used: u64,
    logs_bloom: Bloom,
    logs: Vec<LogT>,
    deposit_nonce: Option<u64>,
    deposit_receipt_version: Option<u8>,
}

impl<LogT> From<Eip658OrDeposit<LogT>> for Execution<LogT> {
    fn from(value: Eip658OrDeposit<LogT>) -> Self {
        if let Some(deposit_nonce) = value.deposit_nonce {
            Execution::Deposit(Deposit {
                status: value.status,
                cumulative_gas_used: value.cumulative_gas_used,
                logs_bloom: value.logs_bloom,
                logs: value.logs,
                deposit_nonce,
                deposit_receipt_version: value.deposit_receipt_version,
            })
        } else {
            Execution::Eip658(super::Eip658 {
                status: value.status,
                cumulative_gas_used: value.cumulative_gas_used,
                logs_bloom: value.logs_bloom,
                logs: value.logs,
            })
        }
    }
}

#[derive(RlpEncodable)]
#[rlp(trailing)]
struct Encodable<'receipt, LogT> {
    status: bool,
    cumulative_gas_used: u64,
    logs_bloom: &'receipt Bloom,
    logs: &'receipt Vec<LogT>,
    deposit_nonce: Option<u64>,
    deposit_receipt_version: Option<u8>,
}

impl<'receipt, LogT> From<&'receipt Deposit<LogT>> for Encodable<'receipt, LogT> {
    fn from(receipt: &'receipt Deposit<LogT>) -> Self {
        Self {
            status: receipt.status,
            cumulative_gas_used: receipt.cumulative_gas_used,
            logs_bloom: &receipt.logs_bloom,
            logs: &receipt.logs,
            // Before Canyon, `deposit_nonce` is not present in the RLP-encoding of a receipt.
            // `deposit_receipt_version` is only present post-Canyon, so we can use it to determine
            // whether we're pre-Canyon.
            // Source: <https://specs.optimism.io/protocol/deposits.html#deposit-receipt>
            deposit_nonce: if receipt.deposit_receipt_version.is_some() {
                Some(receipt.deposit_nonce)
            } else {
                None
            },
            deposit_receipt_version: receipt.deposit_receipt_version,
        }
    }
}

impl<LogT> alloy_rlp::Encodable for Deposit<LogT>
where
    LogT: alloy_rlp::Encodable,
{
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        Encodable::from(self).encode(out);
    }

    fn length(&self) -> usize {
        Encodable::from(self).length()
    }
}

// implement MapReceiptLogs for Deposit
impl<LogT, NewLogT> MapReceiptLogs<LogT, NewLogT, Deposit<NewLogT>> for Deposit<LogT> {
    fn map_logs(self, map_fn: impl FnMut(LogT) -> NewLogT) -> Deposit<NewLogT> {
        Deposit {
            status: self.status,
            cumulative_gas_used: self.cumulative_gas_used,
            logs_bloom: self.logs_bloom,
            logs: self.logs.into_iter().map(map_fn).collect(),
            deposit_nonce: self.deposit_nonce,
            deposit_receipt_version: self.deposit_receipt_version,
        }
    }
}
