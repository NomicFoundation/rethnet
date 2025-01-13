mod deposit;

pub use edr_eth::receipt::execution::{Eip658, Legacy};
use edr_eth::{
    log::ExecutionLog,
    receipt::{ExecutionReceipt, MapReceiptLogs, RootOrStatus},
    result::ExecutionResult,
    transaction::{Transaction as _, TransactionType as _},
    Bloom,
};
use edr_evm::{receipt::ExecutionReceiptBuilder, state::State};
use revm_optimism::{OptimismHaltReason, OptimismSpecId};

use self::deposit::Eip658OrDeposit;
use super::Execution;
use crate::{eip2718::TypedEnvelope, transaction};

/// Receipt for an Optimism deposit transaction with deposit nonce (since
/// Regolith) and optionally deposit receipt version (since Canyon).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deposit<LogT> {
    /// Status
    pub status: bool,
    /// Cumulative gas used in block after this transaction was executed
    pub cumulative_gas_used: u64,
    /// Bloom filter of the logs generated within this transaction
    pub logs_bloom: Bloom,
    /// Logs generated within this transaction
    pub logs: Vec<LogT>,
    /// The nonce used during execution.
    pub deposit_nonce: u64,
    /// The deposit receipt version.
    ///
    /// The deposit receipt version was introduced in Canyon to indicate an
    /// update to how receipt hashes should be computed when set. The state
    /// transition process ensures this is only set for post-Canyon deposit
    /// transactions.
    pub deposit_receipt_version: Option<u8>,
}

impl<LogT> From<Legacy<LogT>> for Execution<LogT> {
    fn from(value: Legacy<LogT>) -> Self {
        Execution::Legacy(value)
    }
}

impl<LogT> From<Eip658<LogT>> for Execution<LogT> {
    fn from(value: Eip658<LogT>) -> Self {
        Execution::Eip658(value)
    }
}

impl<LogT> From<Deposit<LogT>> for Execution<LogT> {
    fn from(value: Deposit<LogT>) -> Self {
        Execution::Deposit(value)
    }
}

impl<LogT> alloy_rlp::Decodable for Execution<LogT>
where
    LogT: alloy_rlp::Decodable,
{
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        // Use a temporary buffer to decode the header, avoiding the original buffer
        // from being advanced
        let first_value_header = {
            let mut temp_buf = *buf;

            let _receipt_header = alloy_rlp::Header::decode(&mut temp_buf)?;
            alloy_rlp::Header::decode(&mut temp_buf)?
        };

        // The first value of the receipt is 1 byte long, which means it's the status
        // code of an EIP-658 receipt.
        if first_value_header.payload_length == 1 {
            let receipt = Eip658OrDeposit::decode(buf)?;
            Ok(receipt.into())
        } else {
            let receipt = Legacy::<LogT>::decode(buf)?;
            Ok(Self::Legacy(receipt))
        }
    }
}

impl<LogT> alloy_rlp::Encodable for Execution<LogT>
where
    LogT: alloy_rlp::Encodable,
{
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        match self {
            Execution::Legacy(receipt) => receipt.encode(out),
            Execution::Eip658(receipt) => receipt.encode(out),
            Execution::Deposit(receipt) => receipt.encode(out),
        }
    }

    fn length(&self) -> usize {
        match self {
            Execution::Legacy(receipt) => receipt.length(),
            Execution::Eip658(receipt) => receipt.length(),
            Execution::Deposit(receipt) => receipt.length(),
        }
    }
}

/// Optimism execution receipt builder.
pub struct Builder {
    deposit_nonce: u64,
}

impl ExecutionReceiptBuilder<OptimismHaltReason, OptimismSpecId, transaction::Signed> for Builder {
    type Receipt = TypedEnvelope<Execution<ExecutionLog>>;

    fn new_receipt_builder<StateT: State>(
        pre_execution_state: StateT,
        transaction: &transaction::Signed,
    ) -> Result<Self, StateT::Error> {
        let deposit_nonce = pre_execution_state
            .basic(*transaction.caller())?
            .map_or(0, |account| account.nonce);

        Ok(Self { deposit_nonce })
    }

    fn build_receipt(
        self,
        header: &edr_eth::block::PartialHeader,
        transaction: &transaction::Signed,
        result: &ExecutionResult<OptimismHaltReason>,
        hardfork: OptimismSpecId,
    ) -> Self::Receipt {
        let logs = result.logs().to_vec();
        let logs_bloom = edr_eth::log::logs_to_bloom(&logs);

        let receipt = if transaction.transaction_type() == transaction::Type::Deposit {
            Execution::Deposit(Deposit {
                status: result.is_success(),
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
                deposit_nonce: self.deposit_nonce,
                deposit_receipt_version: if hardfork >= OptimismSpecId::CANYON {
                    Some(1)
                } else {
                    None
                },
            })
        } else if hardfork >= OptimismSpecId::BYZANTIUM {
            Execution::Eip658(Eip658 {
                status: result.is_success(),
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        } else {
            Execution::Legacy(Legacy {
                root: header.state_root,
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        };

        TypedEnvelope::new(receipt, transaction.transaction_type())
    }
}

impl<LogT, NewLogT> MapReceiptLogs<LogT, NewLogT, Execution<NewLogT>> for Execution<LogT> {
    fn map_logs(self, map_fn: impl FnMut(LogT) -> NewLogT) -> Execution<NewLogT> {
        match self {
            Execution::Legacy(receipt) => Execution::Legacy(receipt.map_logs(map_fn)),
            Execution::Eip658(receipt) => Execution::Eip658(receipt.map_logs(map_fn)),
            Execution::Deposit(receipt) => Execution::Deposit(receipt.map_logs(map_fn)),
        }
    }
}

impl<LogT> ExecutionReceipt for Execution<LogT> {
    type Log = LogT;

    fn cumulative_gas_used(&self) -> u64 {
        match self {
            Execution::Legacy(receipt) => receipt.cumulative_gas_used,
            Execution::Eip658(receipt) => receipt.cumulative_gas_used,
            Execution::Deposit(receipt) => receipt.cumulative_gas_used,
        }
    }

    fn logs_bloom(&self) -> &Bloom {
        match self {
            Execution::Legacy(receipt) => &receipt.logs_bloom,
            Execution::Eip658(receipt) => &receipt.logs_bloom,
            Execution::Deposit(receipt) => &receipt.logs_bloom,
        }
    }

    fn transaction_logs(&self) -> &[LogT] {
        match self {
            Execution::Legacy(receipt) => &receipt.logs,
            Execution::Eip658(receipt) => &receipt.logs,
            Execution::Deposit(receipt) => &receipt.logs,
        }
    }

    fn root_or_status(&self) -> edr_eth::receipt::RootOrStatus<'_> {
        match self {
            Execution::Legacy(receipt) => RootOrStatus::Root(&receipt.root),
            Execution::Eip658(receipt) => RootOrStatus::Status(receipt.status),
            Execution::Deposit(receipt) => RootOrStatus::Status(receipt.status),
        }
    }
}
#[cfg(test)]
mod tests {
    use alloy_rlp::Decodable as _;
    use edr_eth::{log::ExecutionLog, Address, Bytes, B256};

    use super::*;
    use crate::eip2718::TypedEnvelope;

    macro_rules! impl_execution_receipt_tests {
        ($(
            $name:ident => $receipt:expr,
        )+) => {
            $(
                paste::item! {
                    #[test]
                    fn [<typed_receipt_rlp_encoding_ $name>]() -> anyhow::Result<()> {
                        let receipt = $receipt;

                        let encoded = alloy_rlp::encode(&receipt);
                        let decoded = TypedEnvelope::<Execution::<ExecutionLog>>::decode(&mut encoded.as_slice())?;
                        assert_eq!(decoded, receipt);

                        Ok(())
                    }
                }
            )+
        };
    }

    impl_execution_receipt_tests! {
        legacy => TypedEnvelope::Legacy(Execution::Legacy(Legacy {
            root: B256::random(),
            cumulative_gas_used: 0xffff,
            logs_bloom: Bloom::random(),
            logs: vec![
                ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
            ],
        })),
        eip658_eip2930 => TypedEnvelope::Eip2930(Execution::Eip658(Eip658 {
            status: true,
            cumulative_gas_used: 0xffff,
            logs_bloom: Bloom::random(),
            logs: vec![
                ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
            ],
        })),
        eip658_eip1559 => TypedEnvelope::Eip2930(Execution::Eip658(Eip658 {
            status: true,
            cumulative_gas_used: 0xffff,
            logs_bloom: Bloom::random(),
            logs: vec![
                ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
            ],
        })),
        eip658_eip4844 => TypedEnvelope::Eip4844(Execution::Eip658(Eip658 {
            status: true,
            cumulative_gas_used: 0xffff,
            logs_bloom: Bloom::random(),
            logs: vec![
                ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
            ],
        })),
        deposit => TypedEnvelope::Deposit(Execution::Deposit(Deposit {
            status: true,
            cumulative_gas_used: 0xffff,
            logs_bloom: Bloom::random(),
            logs: vec![
                ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                ExecutionLog::new_unchecked(Address::random(), Vec::new(),Bytes::from_static(b"test")),
            ],
            deposit_nonce: 0x1234,
            deposit_receipt_version: Some(0x01),
        })),
    }
}
