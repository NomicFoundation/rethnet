use alloy_rlp::Buf as _;
use edr_eth::{
    receipt::{ExecutionReceipt, MapReceiptLogs, RootOrStatus},
    transaction::TransactionType,
    Bloom,
};

use crate::transaction;

/// An compile-time typed EIP-2718 envelope for L1 Ethereum.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypedEnvelope<DataT> {
    /// Legacy transaction.
    Legacy(DataT),
    /// EIP-2930 transaction.
    Eip2930(DataT),
    /// EIP-1559 transaction.
    Eip1559(DataT),
    /// EIP-4844 transaction.
    Eip4844(DataT),
    /// Optimism deposit transaction
    Deposit(DataT),
}

impl<DataT> TypedEnvelope<DataT> {
    /// Constructs a typed envelope around the given data.
    pub fn new(data: DataT, transaction_type: transaction::Type) -> Self {
        match transaction_type {
            transaction::Type::Legacy => Self::Legacy(data),
            transaction::Type::Eip2930 => Self::Eip2930(data),
            transaction::Type::Eip1559 => Self::Eip1559(data),
            transaction::Type::Eip4844 => Self::Eip4844(data),
            transaction::Type::Deposit => Self::Deposit(data),
        }
    }

    /// Returns a reference to the data inside the envelope.
    pub fn data(&self) -> &DataT {
        match self {
            TypedEnvelope::Legacy(data)
            | TypedEnvelope::Eip2930(data)
            | TypedEnvelope::Eip1559(data)
            | TypedEnvelope::Eip4844(data)
            | TypedEnvelope::Deposit(data) => data,
        }
    }

    /// Maps the data inside the envelope to a new type.
    pub fn map<NewDataT, F>(self, f: F) -> TypedEnvelope<NewDataT>
    where
        F: FnOnce(DataT) -> NewDataT,
    {
        match self {
            TypedEnvelope::Legacy(data) => TypedEnvelope::Legacy(f(data)),
            TypedEnvelope::Eip2930(data) => TypedEnvelope::Eip2930(f(data)),
            TypedEnvelope::Eip1559(data) => TypedEnvelope::Eip1559(f(data)),
            TypedEnvelope::Eip4844(data) => TypedEnvelope::Eip4844(f(data)),
            TypedEnvelope::Deposit(data) => TypedEnvelope::Deposit(f(data)),
        }
    }
}

impl<DataT> TransactionType for TypedEnvelope<DataT> {
    type Type = transaction::Type;

    fn transaction_type(&self) -> Self::Type {
        match self {
            TypedEnvelope::Legacy(_) => transaction::Type::Legacy,
            TypedEnvelope::Eip2930(_) => transaction::Type::Eip2930,
            TypedEnvelope::Eip1559(_) => transaction::Type::Eip1559,
            TypedEnvelope::Eip4844(_) => transaction::Type::Eip4844,
            TypedEnvelope::Deposit(_) => transaction::Type::Deposit,
        }
    }
}

impl<DataT> alloy_rlp::Decodable for TypedEnvelope<DataT>
where
    DataT: alloy_rlp::Decodable,
{
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        fn is_list(byte: u8) -> bool {
            byte >= 0xc0
        }

        let first = *buf.first().ok_or(alloy_rlp::Error::InputTooShort)?;
        let transaction_type = if is_list(first) {
            transaction::Type::Legacy
        } else {
            // Consume the first byte
            buf.advance(1);

            crate::transaction::Type::try_from(first)
                .map_err(|_error| alloy_rlp::Error::Custom("unknown receipt type"))?
        };

        let data = DataT::decode(buf)?;
        Ok(TypedEnvelope::new(data, transaction_type))
    }
}

impl<DataT> alloy_rlp::Encodable for TypedEnvelope<DataT>
where
    DataT: alloy_rlp::Encodable,
{
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        let transaction_type: u8 = self.transaction_type().into();
        if transaction_type > 0 {
            out.put_u8(transaction_type);
        }

        self.data().encode(out);
    }

    fn length(&self) -> usize {
        let type_length = usize::from(u8::from(self.transaction_type()) > 0u8);
        type_length + self.data().length()
    }
}

impl<DataT: ExecutionReceipt> ExecutionReceipt for TypedEnvelope<DataT> {
    type Log = DataT::Log;

    fn cumulative_gas_used(&self) -> u64 {
        self.data().cumulative_gas_used()
    }

    fn logs_bloom(&self) -> &Bloom {
        self.data().logs_bloom()
    }

    fn transaction_logs(&self) -> &[Self::Log] {
        self.data().transaction_logs()
    }

    fn root_or_status(&self) -> RootOrStatus<'_> {
        self.data().root_or_status()
    }
}

impl<OldDataT: MapReceiptLogs<OldLogT, NewLogT, NewDataT>, OldLogT, NewLogT, NewDataT>
    MapReceiptLogs<OldLogT, NewLogT, TypedEnvelope<NewDataT>> for TypedEnvelope<OldDataT>
{
    fn map_logs(self, map_fn: impl FnMut(OldLogT) -> NewLogT) -> TypedEnvelope<NewDataT> {
        self.map(|data| data.map_logs(map_fn))
    }
}
