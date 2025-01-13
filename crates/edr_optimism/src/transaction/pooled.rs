pub use edr_eth::transaction::pooled::{Eip155, Eip1559, Eip2930, Eip4844, Legacy};
use edr_eth::{
    eips::{eip2930, eip7702},
    transaction::{
        signed::PreOrPostEip155, ExecutableTransaction, IsEip155, Transaction, TxKind,
        INVALID_TX_TYPE_ERROR_MESSAGE,
    },
    utils::enveloped,
    Address, Blob, Bytes, B256, U256,
};
use edr_provider::spec::HardforkValidationData;

use super::{Pooled, Signed};

/// An Optimism deposit pooled transaction.
pub type Deposit = super::signed::Deposit;

impl Pooled {
    /// Converts the pooled transaction into a signed transaction.
    pub fn into_payload(self) -> Signed {
        match self {
            Pooled::PreEip155Legacy(tx) => Signed::PreEip155Legacy(tx),
            Pooled::PostEip155Legacy(tx) => Signed::PostEip155Legacy(tx),
            Pooled::Eip2930(tx) => Signed::Eip2930(tx),
            Pooled::Eip1559(tx) => Signed::Eip1559(tx),
            Pooled::Eip4844(tx) => Signed::Eip4844(tx.into_payload()),
            Pooled::Deposit(tx) => Signed::Deposit(tx),
        }
    }
}

impl alloy_rlp::Decodable for Pooled {
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        use alloy_rlp::Buf;

        fn is_list(byte: u8) -> bool {
            byte >= 0xc0
        }

        let first = buf.first().ok_or(alloy_rlp::Error::InputTooShort)?;

        match *first {
            Eip2930::TYPE => {
                buf.advance(1);

                Ok(Pooled::Eip2930(Eip2930::decode(buf)?))
            }
            Eip1559::TYPE => {
                buf.advance(1);

                Ok(Pooled::Eip1559(Eip1559::decode(buf)?))
            }
            Eip4844::TYPE => {
                buf.advance(1);

                Ok(Pooled::Eip4844(Eip4844::decode(buf)?))
            }
            Deposit::TYPE => {
                buf.advance(1);

                Ok(Pooled::Deposit(Deposit::decode(buf)?))
            }
            byte if is_list(byte) => {
                let transaction = PreOrPostEip155::decode(buf)?;
                Ok(transaction.into())
            }
            _ => Err(alloy_rlp::Error::Custom(INVALID_TX_TYPE_ERROR_MESSAGE)),
        }
    }
}

impl alloy_rlp::Encodable for Pooled {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.encode(out),
            Pooled::PostEip155Legacy(tx) => tx.encode(out),
            Pooled::Eip2930(tx) => enveloped(1, tx, out),
            Pooled::Eip1559(tx) => enveloped(2, tx, out),
            Pooled::Eip4844(tx) => enveloped(3, tx, out),
            Pooled::Deposit(tx) => enveloped(4, tx, out),
        }
    }

    fn length(&self) -> usize {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.length(),
            Pooled::PostEip155Legacy(tx) => tx.length(),
            Pooled::Eip2930(tx) => tx.length() + 1,
            Pooled::Eip1559(tx) => tx.length() + 1,
            Pooled::Eip4844(tx) => tx.length() + 1,
            Pooled::Deposit(tx) => tx.length() + 1,
        }
    }
}

impl ExecutableTransaction for Pooled {
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            Pooled::PostEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            Pooled::Eip2930(tx) => tx.effective_gas_price(block_base_fee),
            Pooled::Eip1559(tx) => tx.effective_gas_price(block_base_fee),
            Pooled::Eip4844(tx) => tx.effective_gas_price(block_base_fee),
            Pooled::Deposit(tx) => tx.effective_gas_price(block_base_fee),
        }
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.max_fee_per_gas(),
            Pooled::PostEip155Legacy(tx) => tx.max_fee_per_gas(),
            Pooled::Eip2930(tx) => tx.max_fee_per_gas(),
            Pooled::Eip1559(tx) => tx.max_fee_per_gas(),
            Pooled::Eip4844(tx) => tx.max_fee_per_gas(),
            Pooled::Deposit(tx) => tx.max_fee_per_gas(),
        }
    }

    fn rlp_encoding(&self) -> &Bytes {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.rlp_encoding(),
            Pooled::PostEip155Legacy(tx) => tx.rlp_encoding(),
            Pooled::Eip2930(tx) => tx.rlp_encoding(),
            Pooled::Eip1559(tx) => tx.rlp_encoding(),
            Pooled::Eip4844(tx) => tx.rlp_encoding(),
            Pooled::Deposit(tx) => tx.rlp_encoding(),
        }
    }

    fn total_blob_gas(&self) -> Option<u64> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.total_blob_gas(),
            Pooled::PostEip155Legacy(tx) => tx.total_blob_gas(),
            Pooled::Eip2930(tx) => tx.total_blob_gas(),
            Pooled::Eip1559(tx) => tx.total_blob_gas(),
            Pooled::Eip4844(tx) => tx.total_blob_gas(),
            Pooled::Deposit(tx) => tx.total_blob_gas(),
        }
    }

    fn transaction_hash(&self) -> &B256 {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.transaction_hash(),
            Pooled::PostEip155Legacy(tx) => tx.transaction_hash(),
            Pooled::Eip2930(tx) => tx.transaction_hash(),
            Pooled::Eip1559(tx) => tx.transaction_hash(),
            Pooled::Eip4844(tx) => tx.transaction_hash(),
            Pooled::Deposit(tx) => tx.transaction_hash(),
        }
    }
}

impl From<Pooled> for Signed {
    fn from(value: Pooled) -> Self {
        value.into_payload()
    }
}

impl From<PreOrPostEip155> for Pooled {
    fn from(value: PreOrPostEip155) -> Self {
        match value {
            PreOrPostEip155::Pre(tx) => Self::PreEip155Legacy(tx),
            PreOrPostEip155::Post(tx) => Self::PostEip155Legacy(tx),
        }
    }
}

impl HardforkValidationData for Pooled {
    fn to(&self) -> Option<&Address> {
        Some(self.caller())
    }

    fn gas_price(&self) -> Option<&U256> {
        match self {
            Pooled::PreEip155Legacy(tx) => Some(&tx.gas_price),
            Pooled::PostEip155Legacy(tx) => Some(&tx.gas_price),
            Pooled::Eip2930(tx) => Some(&tx.gas_price),
            Pooled::Eip1559(_) | Pooled::Eip4844(_) | Pooled::Deposit(_) => None,
        }
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        ExecutableTransaction::max_fee_per_gas(self)
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        Transaction::max_priority_fee_per_gas(self)
    }

    fn access_list(&self) -> Option<&Vec<eip2930::AccessListItem>> {
        match self {
            Pooled::PreEip155Legacy(_) | Pooled::PostEip155Legacy(_) | Pooled::Deposit(_) => None,
            Pooled::Eip2930(tx) => Some(tx.access_list.0.as_ref()),
            Pooled::Eip1559(tx) => Some(tx.access_list.0.as_ref()),
            Pooled::Eip4844(tx) => Some(&tx.payload().access_list),
        }
    }

    fn blobs(&self) -> Option<&Vec<Blob>> {
        match self {
            Pooled::Eip4844(tx) => Some(tx.blobs_ref()),
            _ => None,
        }
    }

    fn blob_hashes(&self) -> Option<&Vec<B256>> {
        match self {
            Pooled::Eip4844(tx) => Some(&tx.payload().blob_hashes),
            _ => None,
        }
    }
}

impl IsEip155 for Pooled {
    fn is_eip155(&self) -> bool {
        matches!(self, Pooled::PostEip155Legacy(_))
    }
}

impl Transaction for Pooled {
    fn caller(&self) -> &Address {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.caller(),
            Pooled::PostEip155Legacy(tx) => tx.caller(),
            Pooled::Eip2930(tx) => tx.caller(),
            Pooled::Eip1559(tx) => tx.caller(),
            Pooled::Eip4844(tx) => tx.caller(),
            Pooled::Deposit(tx) => tx.caller(),
        }
    }

    fn gas_limit(&self) -> u64 {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.gas_limit(),
            Pooled::PostEip155Legacy(tx) => tx.gas_limit(),
            Pooled::Eip2930(tx) => tx.gas_limit(),
            Pooled::Eip1559(tx) => tx.gas_limit(),
            Pooled::Eip4844(tx) => tx.gas_limit(),
            Pooled::Deposit(tx) => tx.gas_limit(),
        }
    }

    fn gas_price(&self) -> &U256 {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.gas_price(),
            Pooled::PostEip155Legacy(tx) => tx.gas_price(),
            Pooled::Eip2930(tx) => tx.gas_price(),
            Pooled::Eip1559(tx) => tx.gas_price(),
            Pooled::Eip4844(tx) => tx.gas_price(),
            Pooled::Deposit(tx) => tx.gas_price(),
        }
    }

    fn kind(&self) -> TxKind {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.kind(),
            Pooled::PostEip155Legacy(tx) => tx.kind(),
            Pooled::Eip2930(tx) => tx.kind(),
            Pooled::Eip1559(tx) => tx.kind(),
            Pooled::Eip4844(tx) => tx.kind(),
            Pooled::Deposit(tx) => tx.kind(),
        }
    }

    fn value(&self) -> &U256 {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.value(),
            Pooled::PostEip155Legacy(tx) => tx.value(),
            Pooled::Eip2930(tx) => tx.value(),
            Pooled::Eip1559(tx) => tx.value(),
            Pooled::Eip4844(tx) => tx.value(),
            Pooled::Deposit(tx) => tx.value(),
        }
    }

    fn data(&self) -> &Bytes {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.data(),
            Pooled::PostEip155Legacy(tx) => tx.data(),
            Pooled::Eip2930(tx) => tx.data(),
            Pooled::Eip1559(tx) => tx.data(),
            Pooled::Eip4844(tx) => tx.data(),
            Pooled::Deposit(tx) => tx.data(),
        }
    }

    fn nonce(&self) -> u64 {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.nonce(),
            Pooled::PostEip155Legacy(tx) => tx.nonce(),
            Pooled::Eip2930(tx) => tx.nonce(),
            Pooled::Eip1559(tx) => tx.nonce(),
            Pooled::Eip4844(tx) => tx.nonce(),
            Pooled::Deposit(tx) => tx.nonce(),
        }
    }

    fn chain_id(&self) -> Option<u64> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.chain_id(),
            Pooled::PostEip155Legacy(tx) => tx.chain_id(),
            Pooled::Eip2930(tx) => tx.chain_id(),
            Pooled::Eip1559(tx) => tx.chain_id(),
            Pooled::Eip4844(tx) => tx.chain_id(),
            Pooled::Deposit(tx) => tx.chain_id(),
        }
    }

    fn access_list(&self) -> &[eip2930::AccessListItem] {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.access_list(),
            Pooled::PostEip155Legacy(tx) => tx.access_list(),
            Pooled::Eip2930(tx) => tx.access_list(),
            Pooled::Eip1559(tx) => tx.access_list(),
            Pooled::Eip4844(tx) => tx.access_list(),
            Pooled::Deposit(tx) => tx.access_list(),
        }
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            Pooled::PostEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            Pooled::Eip2930(tx) => tx.max_priority_fee_per_gas(),
            Pooled::Eip1559(tx) => tx.max_priority_fee_per_gas(),
            Pooled::Eip4844(tx) => tx.max_priority_fee_per_gas(),
            Pooled::Deposit(tx) => tx.max_priority_fee_per_gas(),
        }
    }

    fn blob_hashes(&self) -> &[B256] {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.blob_hashes(),
            Pooled::PostEip155Legacy(tx) => tx.blob_hashes(),
            Pooled::Eip2930(tx) => tx.blob_hashes(),
            Pooled::Eip1559(tx) => tx.blob_hashes(),
            Pooled::Eip4844(tx) => tx.blob_hashes(),
            Pooled::Deposit(tx) => tx.blob_hashes(),
        }
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            Pooled::PostEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            Pooled::Eip2930(tx) => tx.max_fee_per_blob_gas(),
            Pooled::Eip1559(tx) => tx.max_fee_per_blob_gas(),
            Pooled::Eip4844(tx) => tx.max_fee_per_blob_gas(),
            Pooled::Deposit(tx) => tx.max_fee_per_blob_gas(),
        }
    }

    fn authorization_list(&self) -> Option<&eip7702::AuthorizationList> {
        match self {
            Pooled::PreEip155Legacy(tx) => tx.authorization_list(),
            Pooled::PostEip155Legacy(tx) => tx.authorization_list(),
            Pooled::Eip2930(tx) => tx.authorization_list(),
            Pooled::Eip1559(tx) => tx.authorization_list(),
            Pooled::Eip4844(tx) => tx.authorization_list(),
            Pooled::Deposit(tx) => tx.authorization_list(),
        }
    }
}
