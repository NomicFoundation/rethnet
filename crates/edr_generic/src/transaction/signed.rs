use edr_eth::{
    eips::{eip2930, eip7702},
    signature::Signature,
    transaction::{
        self, ExecutableTransaction, IsSupported, SignedTransaction, Transaction, TransactionMut,
        TransactionType, TransactionValidation, TxKind,
    },
    Address, Bytes, B256, U256,
};

/// The type of transaction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Type {
    #[default]
    /// Legacy transaction
    Legacy = transaction::signed::Legacy::TYPE,
    /// EIP-2930 transaction
    Eip2930 = transaction::signed::Eip2930::TYPE,
    /// EIP-1559 transaction
    Eip1559 = transaction::signed::Eip1559::TYPE,
    /// EIP-4844 transaction
    Eip4844 = transaction::signed::Eip4844::TYPE,
    /// Unrecognized transaction type.
    Unrecognized(u8),
}

impl From<Type> for u8 {
    fn from(t: Type) -> u8 {
        match t {
            Type::Legacy => transaction::signed::Legacy::TYPE,
            Type::Eip2930 => transaction::signed::Eip2930::TYPE,
            Type::Eip1559 => transaction::signed::Eip1559::TYPE,
            Type::Eip4844 => transaction::signed::Eip4844::TYPE,
            Type::Unrecognized(t) => t,
        }
    }
}

impl From<u8> for Type {
    fn from(t: u8) -> Self {
        match t {
            transaction::signed::Legacy::TYPE => Self::Legacy,
            transaction::signed::Eip2930::TYPE => Self::Eip2930,
            transaction::signed::Eip1559::TYPE => Self::Eip1559,
            transaction::signed::Eip4844::TYPE => Self::Eip4844,
            t => Self::Unrecognized(t),
        }
    }
}

impl From<edr_eth::transaction::Type> for Type {
    fn from(t: edr_eth::transaction::Type) -> Self {
        match t {
            edr_eth::transaction::Type::Legacy => Self::Legacy,
            edr_eth::transaction::Type::Eip2930 => Self::Eip2930,
            edr_eth::transaction::Type::Eip1559 => Self::Eip1559,
            edr_eth::transaction::Type::Eip4844 => Self::Eip4844,
        }
    }
}

impl transaction::IsEip4844 for Type {
    fn is_eip4844(&self) -> bool {
        matches!(self, Type::Eip4844)
    }
}

impl transaction::IsLegacy for Type {
    fn is_legacy(&self) -> bool {
        matches!(self, Type::Legacy)
    }
}

/// A regular [`Signed`](edr_eth::transaction::Signed) transaction that falls
/// back to post-EIP 155 legacy transactions for unrecognized transaction types
/// when converting from an RPC request.
// NOTE: This is a newtype only because we need to use a different
// `TryFrom<TransactionWithSignature>` impl that treats unrecognized transaction
// types different.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SignedWithFallbackToPostEip155 {
    inner: transaction::Signed,
    r#type: Type,
}

impl SignedWithFallbackToPostEip155 {
    /// Constructs a new instance with the provided transaction its type.
    pub fn with_type(inner: transaction::Signed, r#type: Type) -> Self {
        Self { inner, r#type }
    }
}

impl alloy_rlp::Encodable for SignedWithFallbackToPostEip155 {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        self.inner.encode(out);
    }

    fn length(&self) -> usize {
        self.inner.length()
    }
}

impl From<transaction::Signed> for SignedWithFallbackToPostEip155 {
    fn from(value: transaction::Signed) -> Self {
        Self {
            r#type: value.transaction_type().into(),
            inner: value,
        }
    }
}

impl IsSupported for SignedWithFallbackToPostEip155 {
    fn is_supported_transaction(&self) -> bool {
        !matches!(self.r#type, Type::Unrecognized(_))
    }
}

impl From<edr_eth::transaction::pooled::PooledTransaction> for SignedWithFallbackToPostEip155 {
    fn from(value: edr_eth::transaction::pooled::PooledTransaction) -> Self {
        edr_eth::transaction::Signed::from(value).into()
    }
}

impl TransactionValidation for SignedWithFallbackToPostEip155 {
    type ValidationError = <transaction::Signed as TransactionValidation>::ValidationError;
}

impl Transaction for SignedWithFallbackToPostEip155 {
    fn caller(&self) -> &Address {
        self.inner.caller()
    }

    fn gas_limit(&self) -> u64 {
        self.inner.gas_limit()
    }

    fn gas_price(&self) -> &U256 {
        self.inner.gas_price()
    }

    fn kind(&self) -> TxKind {
        self.inner.kind()
    }

    fn value(&self) -> &U256 {
        self.inner.value()
    }

    fn data(&self) -> &Bytes {
        self.inner.data()
    }

    fn nonce(&self) -> u64 {
        self.inner.nonce()
    }

    fn chain_id(&self) -> Option<u64> {
        self.inner.chain_id()
    }

    fn access_list(&self) -> &[eip2930::AccessListItem] {
        self.inner.access_list()
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        self.inner.max_priority_fee_per_gas()
    }

    fn blob_hashes(&self) -> &[B256] {
        self.inner.blob_hashes()
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        self.inner.max_fee_per_blob_gas()
    }

    fn authorization_list(&self) -> Option<&eip7702::AuthorizationList> {
        self.inner.authorization_list()
    }
}

impl TransactionMut for SignedWithFallbackToPostEip155 {
    fn set_gas_limit(&mut self, gas_limit: u64) {
        self.inner.set_gas_limit(gas_limit);
    }
}

impl ExecutableTransaction for SignedWithFallbackToPostEip155 {
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256> {
        self.inner.effective_gas_price(block_base_fee)
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        self.inner.max_fee_per_gas()
    }

    fn rlp_encoding(&self) -> &Bytes {
        self.inner.rlp_encoding()
    }

    fn total_blob_gas(&self) -> Option<u64> {
        self.inner.total_blob_gas()
    }

    fn transaction_hash(&self) -> &B256 {
        self.inner.transaction_hash()
    }
}

impl SignedTransaction for SignedWithFallbackToPostEip155 {
    fn signature(&self) -> &dyn Signature {
        self.inner.signature()
    }
}

impl TransactionType for SignedWithFallbackToPostEip155 {
    type Type = Type;

    fn transaction_type(&self) -> Self::Type {
        self.r#type
    }
}

impl transaction::HasAccessList for SignedWithFallbackToPostEip155 {
    fn has_access_list(&self) -> bool {
        self.inner.has_access_list()
    }
}

impl transaction::IsEip155 for SignedWithFallbackToPostEip155 {
    fn is_eip155(&self) -> bool {
        self.inner.is_eip155()
    }
}

impl transaction::IsEip4844 for SignedWithFallbackToPostEip155 {
    fn is_eip4844(&self) -> bool {
        self.inner.is_eip4844()
    }
}

impl transaction::IsLegacy for SignedWithFallbackToPostEip155 {
    fn is_legacy(&self) -> bool {
        self.inner.is_legacy()
    }
}
