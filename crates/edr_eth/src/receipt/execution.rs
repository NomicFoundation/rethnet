mod eip658;
mod legacy;

use alloy_rlp::{RlpDecodable, RlpEncodable};

use super::{Execution, ExecutionReceipt, MapReceiptLogs};
use crate::{Bloom, B256};

#[derive(Clone, Debug, PartialEq, Eq, RlpDecodable, RlpEncodable)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "camelCase")
)]
pub struct Legacy<LogT> {
    /// State root
    pub root: B256,
    /// Cumulative gas used in block after this transaction was executed
    #[cfg_attr(feature = "serde", serde(with = "crate::serde::u64"))]
    pub cumulative_gas_used: u64,
    /// Bloom filter of the logs generated within this transaction
    pub logs_bloom: Bloom,
    /// Logs generated within this transaction
    pub logs: Vec<LogT>,
}

#[derive(Clone, Debug, PartialEq, Eq, RlpDecodable, RlpEncodable)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "camelCase")
)]
pub struct Eip658<LogT> {
    /// Status
    #[cfg_attr(feature = "serde", serde(with = "crate::serde::bool"))]
    pub status: bool,
    /// Cumulative gas used in block after this transaction was executed
    #[cfg_attr(feature = "serde", serde(with = "crate::serde::u64"))]
    pub cumulative_gas_used: u64,
    /// Bloom filter of the logs generated within this transaction
    pub logs_bloom: Bloom,
    /// Logs generated within this transaction
    pub logs: Vec<LogT>,
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

impl<LogT, NewLogT> MapReceiptLogs<LogT, NewLogT, Execution<NewLogT>> for Execution<LogT> {
    fn map_logs(self, map_fn: impl FnMut(LogT) -> NewLogT) -> Execution<NewLogT> {
        match self {
            Execution::Legacy(receipt) => Execution::Legacy(receipt.map_logs(map_fn)),
            Execution::Eip658(receipt) => Execution::Eip658(receipt.map_logs(map_fn)),
        }
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
            let receipt = Eip658::<LogT>::decode(buf)?;
            Ok(Self::Eip658(receipt))
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
        }
    }

    fn length(&self) -> usize {
        match self {
            Execution::Legacy(receipt) => receipt.length(),
            Execution::Eip658(receipt) => receipt.length(),
        }
    }
}

impl<LogT> ExecutionReceipt for Execution<LogT> {
    type Log = LogT;

    fn cumulative_gas_used(&self) -> u64 {
        match self {
            Execution::Legacy(receipt) => receipt.cumulative_gas_used,
            Execution::Eip658(receipt) => receipt.cumulative_gas_used,
        }
    }

    fn logs_bloom(&self) -> &Bloom {
        match self {
            Execution::Legacy(receipt) => &receipt.logs_bloom,
            Execution::Eip658(receipt) => &receipt.logs_bloom,
        }
    }

    fn transaction_logs(&self) -> &[LogT] {
        match self {
            Execution::Legacy(receipt) => &receipt.logs,
            Execution::Eip658(receipt) => &receipt.logs,
        }
    }

    fn root_or_status(&self) -> super::RootOrStatus<'_> {
        match self {
            Execution::Legacy(receipt) => super::RootOrStatus::Root(&receipt.root),
            Execution::Eip658(receipt) => super::RootOrStatus::Status(receipt.status),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloy_rlp::Decodable as _;

    use super::*;
    use crate::{eips::eip2718::TypedEnvelope, log::ExecutionLog, Address, Bytes};

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
    }
}
