// Part of this code was inspired by foundry. For the original context see:
// https://github.com/foundry-rs/foundry/blob/01b16238ff87dc7ca8ee3f5f13e389888c2a2ee4/anvil/core/src/eth/transaction/mod.rs
#![allow(missing_docs)]

//! transaction related data

mod fake_signature;
/// Types for transaction gossip (aka pooled transactions)
pub mod pooled;
/// Types for transaction requests.
pub mod request;
/// Types for signed transactions.
pub mod signed;

use std::str::FromStr;

pub use revm_primitives::alloy_primitives::TxKind;
use revm_primitives::{ruint, B256};
pub use revm_wiring::{result::InvalidTransaction, Transaction, TransactionValidation};

use crate::{signature::Signature, Bytes, U256, U8};

pub const INVALID_TX_TYPE_ERROR_MESSAGE: &str = "invalid tx type";

/// Container type for various Ethereum transaction requests
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Request {
    /// A legacy transaction request
    Legacy(request::Legacy),
    /// An EIP-155 transaction request
    Eip155(request::Eip155),
    /// An EIP-2930 transaction request
    Eip2930(request::Eip2930),
    /// An EIP-1559 transaction request
    Eip1559(request::Eip1559),
    /// An EIP-4844 transaction request
    Eip4844(request::Eip4844),
}

/// Container type for various signed Ethereum transactions.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Signed {
    /// Legacy transaction
    PreEip155Legacy(signed::Legacy),
    /// EIP-155 transaction
    PostEip155Legacy(signed::Eip155),
    /// EIP-2930 transaction
    Eip2930(signed::Eip2930),
    /// EIP-1559 transaction
    Eip1559(signed::Eip1559),
    /// EIP-4844 transaction
    Eip4844(signed::Eip4844),
}

/// The type of transaction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Legacy transaction
    Legacy = signed::Legacy::TYPE,
    /// EIP-2930 transaction
    Eip2930 = signed::Eip2930::TYPE,
    /// EIP-1559 transaction
    Eip1559 = signed::Eip1559::TYPE,
    /// EIP-4844 transaction
    Eip4844 = signed::Eip4844::TYPE,
}

impl From<Type> for u8 {
    fn from(t: Type) -> u8 {
        t as u8
    }
}

impl IsEip4844 for Type {
    fn is_eip4844(&self) -> bool {
        matches!(self, Type::Eip4844)
    }
}

impl IsLegacy for Type {
    fn is_legacy(&self) -> bool {
        matches!(self, Type::Legacy)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("{0}")]
    BaseConvertError(ruint::BaseConvertError),
    #[error("Invalid digit: {0}")]
    InvalidDigit(char),
    #[error("Invalid radix. Only hexadecimal is supported.")]
    InvalidRadix,
    #[error("Unknown transaction type: {0}")]
    UnknownType(u8),
}

impl From<ruint::ParseError> for ParseError {
    fn from(error: ruint::ParseError) -> Self {
        match error {
            ruint::ParseError::InvalidDigit(c) => ParseError::InvalidDigit(c),
            ruint::ParseError::InvalidRadix(_) => ParseError::InvalidRadix,
            ruint::ParseError::BaseConvertError(error) => ParseError::BaseConvertError(error),
        }
    }
}

impl FromStr for Type {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_char_boundary(2) {
            let (prefix, rest) = s.split_at(2);
            if prefix == "0x" {
                let value = U8::from_str_radix(rest, 16)?;

                Type::try_from(value.to::<u8>()).map_err(ParseError::UnknownType)
            } else {
                Err(ParseError::InvalidRadix)
            }
        } else {
            Err(ParseError::InvalidRadix)
        }
    }
}

impl TryFrom<u8> for Type {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            signed::Legacy::TYPE => Ok(Self::Legacy),
            signed::Eip2930::TYPE => Ok(Self::Eip2930),
            signed::Eip1559::TYPE => Ok(Self::Eip1559),
            signed::Eip4844::TYPE => Ok(Self::Eip4844),
            value => Err(value),
        }
    }
}

#[cfg(feature = "serde")]
impl<'deserializer> serde::Deserialize<'deserializer> for Type {
    fn deserialize<D>(deserializer: D) -> Result<Type, D::Error>
    where
        D: serde::Deserializer<'deserializer>,
    {
        let value = U8::deserialize(deserializer)?;
        Type::try_from(value.to::<u8>()).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Type {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        U8::serialize(&U8::from(u8::from(*self)), serializer)
    }
}

/// Trait for information about executable transactions.
pub trait ExecutableTransaction {
    /// The effective gas price of the transaction, calculated using the
    /// provided block base fee. Only applicable for post-EIP-1559 transactions.
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256>;

    /// The maximum fee per gas the sender is willing to pay. Only applicable
    /// for post-EIP-1559 transactions.
    fn max_fee_per_gas(&self) -> Option<&U256>;

    /// The enveloped (EIP-2718) RLP-encoding of the transaction.
    fn rlp_encoding(&self) -> &Bytes;

    /// The total amount of blob gas used by the transaction. Only applicable
    /// for EIP-4844 transactions.
    fn total_blob_gas(&self) -> Option<u64>;

    /// The hash of the transaction.
    fn transaction_hash(&self) -> &B256;
}

/// Trait for transactions that may be signed.
pub trait MaybeSignedTransaction {
    /// Returns the [`Signature`] of the transaction, if any.
    fn maybe_signature(&self) -> Option<&dyn Signature>;
}

/// Trait for transactions that have been signed.
pub trait SignedTransaction {
    /// Returns the [`Signature`] of the transaction.
    fn signature(&self) -> &dyn Signature;
}

impl<TransactionT: SignedTransaction> MaybeSignedTransaction for TransactionT {
    fn maybe_signature(&self) -> Option<&dyn Signature> {
        Some(self.signature())
    }
}

/// Trait for mutable transactions.
pub trait TransactionMut {
    /// Sets the gas limit of the transaction.
    fn set_gas_limit(&mut self, gas_limit: u64);
}

/// Trait for determining the type of a transaction.
pub trait TransactionType {
    /// Type of the transaction.
    type Type: Into<u8>;

    /// Returns the type of the transaction.
    fn transaction_type(&self) -> Self::Type;
}

/// Trait for determining whether a transaction has an access list.
pub trait HasAccessList {
    /// Whether the transaction has an access list.
    fn has_access_list(&self) -> bool;
}

/// Trait for determining whether a transaction is an EIP-155 transaction.
pub trait IsEip155 {
    /// Whether the transaction is an EIP-155 transaction.
    fn is_eip155(&self) -> bool;
}

/// Trait for determining whether a transaction is an EIP-4844 transaction.
pub trait IsEip4844 {
    /// Whether the transaction is an EIP-4844 transaction.
    fn is_eip4844(&self) -> bool;
}

/// Trait for determining whether a transaction is a legacy transaction.
pub trait IsLegacy {
    /// Whether the transaction is a legacy transaction.
    fn is_legacy(&self) -> bool;
}

/// Trait for determining whether a transaction is natively supported by the
/// chain. Unsupported transactions might still be executable, but can have
/// unexpected side effects.
pub trait IsSupported {
    /// Whether the transaction is natively supported.
    fn is_supported_transaction(&self) -> bool;
}

pub fn max_cost(transaction: &impl Transaction) -> U256 {
    U256::from(transaction.gas_limit()).saturating_mul(*transaction.gas_price())
}

pub fn upfront_cost(transaction: &impl Transaction) -> U256 {
    max_cost(transaction).saturating_add(*transaction.value())
}
