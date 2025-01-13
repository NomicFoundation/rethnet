mod eip155;
mod eip1559;
mod eip2930;
mod eip4844;
mod legacy;

use std::sync::OnceLock;

use alloy_rlp::{Buf, BufMut};
use k256::SecretKey;

pub use self::{
    eip155::Eip155,
    eip1559::Eip1559,
    eip2930::Eip2930,
    eip4844::Eip4844,
    legacy::{Legacy, PreOrPostEip155},
};
use super::{
    ExecutableTransaction, HasAccessList, InvalidTransaction, IsEip155, IsEip4844, IsLegacy,
    IsSupported, Signed, SignedTransaction, Transaction, TransactionMut, TransactionType,
    TransactionValidation, TxKind, INVALID_TX_TYPE_ERROR_MESSAGE,
};
use crate::{
    eips::{self, eip7702},
    signature::{Fakeable, Signature, SignatureError},
    Address, Bytes, B256, U256,
};

/// Trait for signing a transaction request with a fake signature.
pub trait FakeSign {
    /// The type of the signed transaction.
    type Signed;

    /// Signs the transaction with a fake signature.
    fn fake_sign(self, sender: Address) -> Self::Signed;
}

pub trait Sign {
    /// The type of the signed transaction.
    type Signed;

    /// Signs the transaction with the provided secret key, belonging to the
    /// provided sender's address.
    ///
    /// # Safety
    ///
    /// The `caller` and `secret_key` must correspond to the same account.
    unsafe fn sign_for_sender_unchecked(
        self,
        secret_key: &SecretKey,
        caller: Address,
    ) -> Result<Self::Signed, SignatureError>;
}

impl Signed {
    /// Whether this is a legacy (pre-EIP-155) transaction.
    pub fn is_pre_eip155(&self) -> bool {
        matches!(self, Signed::PreEip155Legacy(_))
    }

    /// Whether this is an EIP-1559 transaction.
    pub fn is_eip1559(&self) -> bool {
        matches!(self, Signed::Eip1559(_))
    }

    /// Whether this is an EIP-2930 transaction.
    pub fn is_eip2930(&self) -> bool {
        matches!(self, Signed::Eip2930(_))
    }

    pub fn as_legacy(&self) -> Option<&self::legacy::Legacy> {
        match self {
            Signed::PreEip155Legacy(tx) => Some(tx),
            _ => None,
        }
    }
}

impl alloy_rlp::Decodable for Signed {
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        fn is_list(byte: u8) -> bool {
            byte >= alloy_rlp::EMPTY_LIST_CODE
        }

        let first = buf.first().ok_or(alloy_rlp::Error::InputTooShort)?;

        match *first {
            Eip2930::TYPE => {
                buf.advance(1);

                Ok(Signed::Eip2930(Eip2930::decode(buf)?))
            }
            Eip1559::TYPE => {
                buf.advance(1);

                Ok(Signed::Eip1559(Eip1559::decode(buf)?))
            }
            Eip4844::TYPE => {
                buf.advance(1);

                Ok(Signed::Eip4844(Eip4844::decode(buf)?))
            }
            byte if is_list(byte) => {
                let transaction = PreOrPostEip155::decode(buf)?;
                Ok(transaction.into())
            }
            _ => Err(alloy_rlp::Error::Custom(INVALID_TX_TYPE_ERROR_MESSAGE)),
        }
    }
}

impl alloy_rlp::Encodable for Signed {
    fn encode(&self, out: &mut dyn BufMut) {
        let encoded = self.rlp_encoding();
        out.put_slice(encoded);
    }

    fn length(&self) -> usize {
        match self {
            Signed::PreEip155Legacy(tx) => tx.length(),
            Signed::PostEip155Legacy(tx) => tx.length(),
            Signed::Eip2930(tx) => tx.length() + 1,
            Signed::Eip1559(tx) => tx.length() + 1,
            Signed::Eip4844(tx) => tx.length() + 1,
        }
    }
}

impl Default for Signed {
    fn default() -> Self {
        // This implementation is necessary to be able to use `revm`'s builder pattern.
        Self::PreEip155Legacy(Legacy {
            nonce: 0,
            gas_price: U256::ZERO,
            gas_limit: u64::MAX,
            kind: TxKind::Call(Address::ZERO), // will do nothing
            value: U256::ZERO,
            input: Bytes::new(),
            signature: Fakeable::fake(Address::ZERO, Some(0)),
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        })
    }
}

impl From<self::legacy::Legacy> for Signed {
    fn from(transaction: self::legacy::Legacy) -> Self {
        Self::PreEip155Legacy(transaction)
    }
}

impl From<self::eip155::Eip155> for Signed {
    fn from(transaction: self::eip155::Eip155) -> Self {
        Self::PostEip155Legacy(transaction)
    }
}

impl From<self::eip2930::Eip2930> for Signed {
    fn from(transaction: self::eip2930::Eip2930) -> Self {
        Self::Eip2930(transaction)
    }
}

impl From<self::eip1559::Eip1559> for Signed {
    fn from(transaction: self::eip1559::Eip1559) -> Self {
        Self::Eip1559(transaction)
    }
}

impl From<self::eip4844::Eip4844> for Signed {
    fn from(transaction: self::eip4844::Eip4844) -> Self {
        Self::Eip4844(transaction)
    }
}

impl From<PreOrPostEip155> for Signed {
    fn from(value: PreOrPostEip155) -> Self {
        match value {
            PreOrPostEip155::Pre(tx) => Self::PreEip155Legacy(tx),
            PreOrPostEip155::Post(tx) => Self::PostEip155Legacy(tx),
        }
    }
}

impl HasAccessList for Signed {
    fn has_access_list(&self) -> bool {
        match self {
            Signed::PreEip155Legacy(_) | Signed::PostEip155Legacy(_) => false,
            Signed::Eip2930(_) | Signed::Eip1559(_) | Signed::Eip4844(_) => true,
        }
    }
}

impl IsEip155 for Signed {
    fn is_eip155(&self) -> bool {
        matches!(self, Signed::PostEip155Legacy(_))
    }
}

impl IsEip4844 for Signed {
    fn is_eip4844(&self) -> bool {
        matches!(self, Signed::Eip4844(_))
    }
}

impl IsSupported for Signed {
    fn is_supported_transaction(&self) -> bool {
        true
    }
}

impl IsLegacy for Signed {
    fn is_legacy(&self) -> bool {
        matches!(
            self,
            Signed::PreEip155Legacy(_) | Signed::PostEip155Legacy(_)
        )
    }
}

impl ExecutableTransaction for Signed {
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            Signed::PostEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            Signed::Eip2930(tx) => tx.effective_gas_price(block_base_fee),
            Signed::Eip1559(tx) => tx.effective_gas_price(block_base_fee),
            Signed::Eip4844(tx) => tx.effective_gas_price(block_base_fee),
        }
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.max_fee_per_gas(),
            Signed::PostEip155Legacy(tx) => tx.max_fee_per_gas(),
            Signed::Eip2930(tx) => tx.max_fee_per_gas(),
            Signed::Eip1559(tx) => tx.max_fee_per_gas(),
            Signed::Eip4844(tx) => tx.max_fee_per_gas(),
        }
    }

    fn rlp_encoding(&self) -> &Bytes {
        match self {
            Signed::PreEip155Legacy(tx) => tx.rlp_encoding(),
            Signed::PostEip155Legacy(tx) => tx.rlp_encoding(),
            Signed::Eip2930(tx) => tx.rlp_encoding(),
            Signed::Eip1559(tx) => tx.rlp_encoding(),
            Signed::Eip4844(tx) => tx.rlp_encoding(),
        }
    }

    fn total_blob_gas(&self) -> Option<u64> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.total_blob_gas(),
            Signed::PostEip155Legacy(tx) => tx.total_blob_gas(),
            Signed::Eip2930(tx) => tx.total_blob_gas(),
            Signed::Eip1559(tx) => tx.total_blob_gas(),
            Signed::Eip4844(tx) => tx.total_blob_gas(),
        }
    }

    fn transaction_hash(&self) -> &B256 {
        match self {
            Signed::PreEip155Legacy(t) => t.transaction_hash(),
            Signed::PostEip155Legacy(t) => t.transaction_hash(),
            Signed::Eip2930(t) => t.transaction_hash(),
            Signed::Eip1559(t) => t.transaction_hash(),
            Signed::Eip4844(t) => t.transaction_hash(),
        }
    }
}

impl SignedTransaction for Signed {
    fn signature(&self) -> &dyn Signature {
        match self {
            Signed::PreEip155Legacy(tx) => &tx.signature,
            Signed::PostEip155Legacy(tx) => &tx.signature,
            Signed::Eip2930(tx) => &tx.signature,
            Signed::Eip1559(tx) => &tx.signature,
            Signed::Eip4844(tx) => &tx.signature,
        }
    }
}

impl Transaction for Signed {
    fn caller(&self) -> &Address {
        match self {
            Signed::PreEip155Legacy(tx) => tx.caller(),
            Signed::PostEip155Legacy(tx) => tx.caller(),
            Signed::Eip2930(tx) => tx.caller(),
            Signed::Eip1559(tx) => tx.caller(),
            Signed::Eip4844(tx) => tx.caller(),
        }
    }

    fn gas_limit(&self) -> u64 {
        match self {
            Signed::PreEip155Legacy(tx) => tx.gas_limit(),
            Signed::PostEip155Legacy(tx) => tx.gas_limit(),
            Signed::Eip2930(tx) => tx.gas_limit(),
            Signed::Eip1559(tx) => tx.gas_limit(),
            Signed::Eip4844(tx) => tx.gas_limit(),
        }
    }

    fn gas_price(&self) -> &U256 {
        match self {
            Signed::PreEip155Legacy(tx) => tx.gas_price(),
            Signed::PostEip155Legacy(tx) => tx.gas_price(),
            Signed::Eip2930(tx) => tx.gas_price(),
            Signed::Eip1559(tx) => tx.gas_price(),
            Signed::Eip4844(tx) => tx.gas_price(),
        }
    }

    fn kind(&self) -> TxKind {
        match self {
            Signed::PreEip155Legacy(tx) => tx.kind(),
            Signed::PostEip155Legacy(tx) => tx.kind(),
            Signed::Eip2930(tx) => tx.kind(),
            Signed::Eip1559(tx) => tx.kind(),
            Signed::Eip4844(tx) => tx.kind(),
        }
    }

    fn value(&self) -> &U256 {
        match self {
            Signed::PreEip155Legacy(tx) => tx.value(),
            Signed::PostEip155Legacy(tx) => tx.value(),
            Signed::Eip2930(tx) => tx.value(),
            Signed::Eip1559(tx) => tx.value(),
            Signed::Eip4844(tx) => tx.value(),
        }
    }

    fn data(&self) -> &Bytes {
        match self {
            Signed::PreEip155Legacy(tx) => tx.data(),
            Signed::PostEip155Legacy(tx) => tx.data(),
            Signed::Eip2930(tx) => tx.data(),
            Signed::Eip1559(tx) => tx.data(),
            Signed::Eip4844(tx) => tx.data(),
        }
    }

    fn nonce(&self) -> u64 {
        match self {
            Signed::PreEip155Legacy(tx) => tx.nonce(),
            Signed::PostEip155Legacy(tx) => tx.nonce(),
            Signed::Eip2930(tx) => tx.nonce(),
            Signed::Eip1559(tx) => tx.nonce(),
            Signed::Eip4844(tx) => tx.nonce(),
        }
    }

    fn chain_id(&self) -> Option<u64> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.chain_id(),
            Signed::PostEip155Legacy(tx) => tx.chain_id(),
            Signed::Eip2930(tx) => tx.chain_id(),
            Signed::Eip1559(tx) => tx.chain_id(),
            Signed::Eip4844(tx) => tx.chain_id(),
        }
    }

    fn access_list(&self) -> &[eips::eip2930::AccessListItem] {
        match self {
            Signed::PreEip155Legacy(tx) => tx.access_list(),
            Signed::PostEip155Legacy(tx) => tx.access_list(),
            Signed::Eip2930(tx) => tx.access_list(),
            Signed::Eip1559(tx) => tx.access_list(),
            Signed::Eip4844(tx) => tx.access_list(),
        }
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            Signed::PostEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            Signed::Eip2930(tx) => tx.max_priority_fee_per_gas(),
            Signed::Eip1559(tx) => tx.max_priority_fee_per_gas(),
            Signed::Eip4844(tx) => tx.max_priority_fee_per_gas(),
        }
    }

    fn blob_hashes(&self) -> &[B256] {
        match self {
            Signed::PreEip155Legacy(tx) => tx.blob_hashes(),
            Signed::PostEip155Legacy(tx) => tx.blob_hashes(),
            Signed::Eip2930(tx) => tx.blob_hashes(),
            Signed::Eip1559(tx) => tx.blob_hashes(),
            Signed::Eip4844(tx) => tx.blob_hashes(),
        }
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            Signed::PostEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            Signed::Eip2930(tx) => tx.max_fee_per_blob_gas(),
            Signed::Eip1559(tx) => tx.max_fee_per_blob_gas(),
            Signed::Eip4844(tx) => tx.max_fee_per_blob_gas(),
        }
    }

    fn authorization_list(&self) -> Option<&eip7702::AuthorizationList> {
        match self {
            Signed::PreEip155Legacy(tx) => tx.authorization_list(),
            Signed::PostEip155Legacy(tx) => tx.authorization_list(),
            Signed::Eip2930(tx) => tx.authorization_list(),
            Signed::Eip1559(tx) => tx.authorization_list(),
            Signed::Eip4844(tx) => tx.authorization_list(),
        }
    }
}

impl TransactionMut for Signed {
    fn set_gas_limit(&mut self, gas_limit: u64) {
        match self {
            Signed::PreEip155Legacy(tx) => tx.gas_limit = gas_limit,
            Signed::PostEip155Legacy(tx) => tx.gas_limit = gas_limit,
            Signed::Eip2930(tx) => tx.gas_limit = gas_limit,
            Signed::Eip1559(tx) => tx.gas_limit = gas_limit,
            Signed::Eip4844(tx) => tx.gas_limit = gas_limit,
        }
    }
}

impl TransactionType for Signed {
    type Type = super::Type;

    fn transaction_type(&self) -> Self::Type {
        match self {
            Signed::PreEip155Legacy(_) | Signed::PostEip155Legacy(_) => super::Type::Legacy,
            Signed::Eip2930(_) => super::Type::Eip2930,
            Signed::Eip1559(_) => super::Type::Eip1559,
            Signed::Eip4844(_) => super::Type::Eip4844,
        }
    }
}

impl TransactionValidation for Signed {
    type ValidationError = InvalidTransaction;
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use alloy_rlp::Decodable as _;

    use super::*;
    use crate::{signature, transaction, Bytes};

    #[test]
    fn can_recover_sender() {
        // Generated based on
        // "f85f800182520894095e7baea6a6c7c4c2dfeb977efac326af552d870a801ba048b55bfa915ac795c431978d8a6a992b628d557da5ff759b307d495a36649353a0efffd310ac743f371de3b9f7f9cb56c0b28ad43601b4ab949f53faa07bd2c804"
        // but with a normalized signature
        let bytes = hex::decode("f85f800182520894095e7baea6a6c7c4c2dfeb977efac326af552d870a801ca048b55bfa915ac795c431978d8a6a992b628d557da5ff759b307d495a36649353a010002cef538bc0c8e21c46080634a93e082408b0ad93f4a7207e63ec5463793d").unwrap();

        let tx = Signed::decode(&mut bytes.as_slice()).expect("decoding TypedTransaction failed");

        let tx = match tx {
            Signed::PreEip155Legacy(tx) => tx,
            _ => panic!("Invalid typed transaction"),
        };

        assert_eq!(tx.input, Bytes::new());
        assert_eq!(tx.gas_price, U256::from(0x01u64));
        assert_eq!(tx.gas_limit, 0x5208u64);
        assert_eq!(tx.nonce, 0x00u64);
        if let TxKind::Call(ref to) = tx.kind {
            assert_eq!(
                *to,
                "0x095e7baea6a6c7c4c2dfeb977efac326af552d87"
                    .parse::<Address>()
                    .unwrap()
            );
        } else {
            panic!();
        }
        assert_eq!(tx.value, U256::from(0x0au64));
        assert_eq!(
            *tx.caller(),
            "0x0f65fe9276bc9a24ae7083ae28e2660ef72df99e"
                .parse::<Address>()
                .unwrap()
        );
    }

    macro_rules! impl_test_signed_transaction_encoding_round_trip {
        ($(
            $name:ident => $request:expr,
        )+) => {
            $(
                paste::item! {
                    #[test]
                    fn [<signed_transaction_encoding_round_trip_ $name>]() -> anyhow::Result<()> {
                        use signature::secret_key_from_str;

                        let request = $request;

                        let secret_key = secret_key_from_str(edr_defaults::SECRET_KEYS[0]).expect("Failed to parse secret key");
                        let transaction = request.sign(&secret_key)?;

                        let transaction = Signed::from(transaction);

                        let encoded = alloy_rlp::encode(&transaction);
                        let decoded = Signed::decode(&mut encoded.as_slice()).unwrap();

                        assert_eq!(decoded, transaction);

                        Ok(())
                    }
                }
            )+
        };
    }

    impl_test_signed_transaction_encoding_round_trip! {
            pre_eip155 => transaction::request::Legacy {
                nonce: 0,
                gas_price: U256::from(1),
                gas_limit: 2,
                kind: TxKind::Call(Address::default()),
                value: U256::from(3),
                input: Bytes::from(vec![1, 2]),
            },
            post_eip155 => transaction::request::Eip155 {
                nonce: 0,
                gas_price: U256::from(1),
                gas_limit: 2,
                kind: TxKind::Create,
                value: U256::from(3),
                input: Bytes::from(vec![1, 2]),
                chain_id: 1337,
            },
            eip2930 => transaction::request::Eip2930 {
                chain_id: 1,
                nonce: 0,
                gas_price: U256::from(1),
                gas_limit: 2,
                kind: TxKind::Call(Address::default()),
                value: U256::from(3),
                input: Bytes::from(vec![1, 2]),
                access_list: vec![],
            },
            eip1559 => transaction::request::Eip1559 {
                chain_id: 1,
                nonce: 0,
                max_priority_fee_per_gas: U256::from(1),
                max_fee_per_gas: U256::from(2),
                gas_limit: 3,
                kind: TxKind::Create,
                value: U256::from(4),
                input: Bytes::from(vec![1, 2]),
                access_list: vec![],
            },
            eip4844 => transaction::request::Eip4844 {
                chain_id: 1,
                nonce: 0,
                max_priority_fee_per_gas: U256::from(1),
                max_fee_per_gas: U256::from(2),
                max_fee_per_blob_gas: U256::from(7),
                gas_limit: 3,
                to: Address::random(),
                value: U256::from(4),
                input: Bytes::from(vec![1, 2]),
                access_list: vec![],
                blob_hashes: vec![B256::random(), B256::random()],
            },
    }

    #[test]
    fn test_signed_transaction_decode_multiple_networks() -> anyhow::Result<()> {
        use std::str::FromStr;

        let bytes_first = hex::decode("f86b02843b9aca00830186a094d3e8763675e4c425df46cc3b5c0f6cbdac39604687038d7ea4c68000802ba00eb96ca19e8a77102767a41fc85a36afd5c61ccb09911cec5d3e86e193d9c5aea03a456401896b1b6055311536bf00a718568c744d8c1f9df59879e8350220ca18").unwrap();
        let expected = Signed::PostEip155Legacy(self::eip155::Eip155 {
            nonce: 2u64,
            gas_price: U256::from(1000000000u64),
            gas_limit: 100000,
            kind: TxKind::Call(Address::from_slice(
                &hex::decode("d3e8763675e4c425df46cc3b5c0f6cbdac396046").unwrap(),
            )),
            value: U256::from(1000000000000000u64),
            input: Bytes::default(),
            // SAFETY: Caller address has been precomputed
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: U256::from_str(
                            "0xeb96ca19e8a77102767a41fc85a36afd5c61ccb09911cec5d3e86e193d9c5ae",
                        )
                        .unwrap(),
                        s: U256::from_str(
                            "0x3a456401896b1b6055311536bf00a718568c744d8c1f9df59879e8350220ca18",
                        )
                        .unwrap(),
                        v: 43,
                    },
                    Address::from_str("0x2efc0b963da6f672254b4e5eea754551fe191fd6")?,
                )
            },
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });
        assert_eq!(
            expected,
            Signed::decode(&mut bytes_first.as_slice()).unwrap()
        );

        let bytes_second = hex::decode("f86b01843b9aca00830186a094d3e8763675e4c425df46cc3b5c0f6cbdac3960468702769bb01b2a00802ba0e24d8bd32ad906d6f8b8d7741e08d1959df021698b19ee232feba15361587d0aa05406ad177223213df262cb66ccbb2f46bfdccfdfbbb5ffdda9e2c02d977631da").unwrap();
        let expected = Signed::PostEip155Legacy(self::eip155::Eip155 {
            nonce: 1,
            gas_price: U256::from(1000000000u64),
            gas_limit: 100000,
            kind: TxKind::Call(Address::from_slice(
                &hex::decode("d3e8763675e4c425df46cc3b5c0f6cbdac396046").unwrap(),
            )),
            value: U256::from(693361000000000u64),
            input: Bytes::default(),
            // SAFETY: Caller address has been precomputed
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: U256::from_str(
                            "0xe24d8bd32ad906d6f8b8d7741e08d1959df021698b19ee232feba15361587d0a",
                        )
                        .unwrap(),
                        s: U256::from_str(
                            "0x5406ad177223213df262cb66ccbb2f46bfdccfdfbbb5ffdda9e2c02d977631da",
                        )
                        .unwrap(),
                        v: 43,
                    },
                    Address::from_str("0x2efc0b963da6f672254b4e5eea754551fe191fd6")?,
                )
            },
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });
        assert_eq!(
            expected,
            Signed::decode(&mut bytes_second.as_slice()).unwrap()
        );

        let bytes_third = hex::decode("f86b0384773594008398968094d3e8763675e4c425df46cc3b5c0f6cbdac39604687038d7ea4c68000802ba0ce6834447c0a4193c40382e6c57ae33b241379c5418caac9cdc18d786fd12071a03ca3ae86580e94550d7c071e3a02eadb5a77830947c9225165cf9100901bee88").unwrap();
        let expected = Signed::PostEip155Legacy(self::eip155::Eip155 {
            nonce: 3,
            gas_price: U256::from(2000000000u64),
            gas_limit: 10000000,
            kind: TxKind::Call(Address::from_slice(
                &hex::decode("d3e8763675e4c425df46cc3b5c0f6cbdac396046").unwrap(),
            )),
            value: U256::from(1000000000000000u64),
            input: Bytes::default(),
            // SAFETY: Caller address has been precomputed
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: U256::from_str(
                            "0xce6834447c0a4193c40382e6c57ae33b241379c5418caac9cdc18d786fd12071",
                        )
                        .unwrap(),
                        s: U256::from_str(
                            "0x3ca3ae86580e94550d7c071e3a02eadb5a77830947c9225165cf9100901bee88",
                        )
                        .unwrap(),
                        v: 43,
                    },
                    Address::from_str("0x2efc0b963da6f672254b4e5eea754551fe191fd6")?,
                )
            },
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });
        assert_eq!(
            expected,
            Signed::decode(&mut bytes_third.as_slice()).unwrap()
        );

        let bytes_fourth = hex::decode("02f872041a8459682f008459682f0d8252089461815774383099e24810ab832a5b2a5425c154d58829a2241af62c000080c001a059e6b67f48fb32e7e570dfb11e042b5ad2e55e3ce3ce9cd989c7e06e07feeafda0016b83f4f980694ed2eee4d10667242b1f40dc406901b34125b008d334d47469").unwrap();
        let expected = Signed::Eip1559(self::eip1559::Eip1559 {
            chain_id: 4,
            nonce: 26,
            max_priority_fee_per_gas: U256::from(1500000000u64),
            max_fee_per_gas: U256::from(1500000013u64),
            gas_limit: 21000,
            kind: TxKind::Call(Address::from_slice(
                &hex::decode("61815774383099e24810ab832a5b2a5425c154d5").unwrap(),
            )),
            value: U256::from(3000000000000000000u64),
            input: Bytes::default(),
            access_list: eips::eip2930::AccessList::default(),
            // SAFETY: Caller address has been precomputed
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithYParity {
                        r: U256::from_str(
                            "0x59e6b67f48fb32e7e570dfb11e042b5ad2e55e3ce3ce9cd989c7e06e07feeafd",
                        )
                        .unwrap(),
                        s: U256::from_str(
                            "0x016b83f4f980694ed2eee4d10667242b1f40dc406901b34125b008d334d47469",
                        )
                        .unwrap(),
                        y_parity: true,
                    },
                    Address::from_str("0x9421de2177f0e810ca1d69a040a2169f8c7c8e4b")?,
                )
            },
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });
        assert_eq!(
            expected,
            Signed::decode(&mut bytes_fourth.as_slice()).unwrap()
        );

        let bytes_fifth = hex::decode("f8650f84832156008287fb94cf7f9e66af820a19257a2108375b180b0ec491678204d2802ca035b7bfeb9ad9ece2cbafaaf8e202e706b4cfaeb233f46198f00b44d4a566a981a0612638fb29427ca33b9a3be2a0a561beecfe0269655be160d35e72d366a6a860").unwrap();
        let expected = Signed::PostEip155Legacy(self::eip155::Eip155 {
            nonce: 15u64,
            gas_price: U256::from(2200000000u64),
            gas_limit: 34811,
            kind: TxKind::Call(Address::from_slice(
                &hex::decode("cf7f9e66af820a19257a2108375b180b0ec49167").unwrap(),
            )),
            value: U256::from(1234u64),
            input: Bytes::default(),
            // SAFETY: Caller address has been precomputed
            signature: unsafe {
                signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithRecoveryId {
                        r: U256::from_str(
                            "0x35b7bfeb9ad9ece2cbafaaf8e202e706b4cfaeb233f46198f00b44d4a566a981",
                        )
                        .unwrap(),
                        s: U256::from_str(
                            "0x612638fb29427ca33b9a3be2a0a561beecfe0269655be160d35e72d366a6a860",
                        )
                        .unwrap(),
                        v: 44,
                    },
                    Address::from_str("0xd35bd31431b33b756f965af3c62776354d6e4bd8")?,
                )
            },
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });
        assert_eq!(
            expected,
            Signed::decode(&mut bytes_fifth.as_slice()).unwrap()
        );

        Ok(())
    }

    // <https://github.com/gakonst/ethers-rs/issues/1732>
    #[test]
    fn test_recover_legacy_tx() {
        let raw_tx = "f9015482078b8505d21dba0083022ef1947a250d5630b4cf539739df2c5dacb4c659f2488d880c46549a521b13d8b8e47ff36ab50000000000000000000000000000000000000000000066ab5a608bd00a23f2fe000000000000000000000000000000000000000000000000000000000000008000000000000000000000000048c04ed5691981c42154c6167398f95e8f38a7ff00000000000000000000000000000000000000000000000000000000632ceac70000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc20000000000000000000000006c6ee5e31d828de241282b9606c8e98ea48526e225a0c9077369501641a92ef7399ff81c21639ed4fd8fc69cb793cfa1dbfab342e10aa0615facb2f1bcf3274a354cfe384a38d0cc008a11c2dd23a69111bc6930ba27a8";

        let tx: Signed = Signed::decode(&mut hex::decode(raw_tx).unwrap().as_slice()).unwrap();
        let expected: Address = "0xa12e1462d0ced572f396f58b6e2d03894cd7c8a4"
            .parse()
            .unwrap();
        assert_eq!(expected, *tx.caller());
    }

    #[test]
    fn from_is_implemented_for_all_variants() {
        fn _compile_test(transaction: Signed) -> Signed {
            match transaction {
                Signed::PreEip155Legacy(transaction) => transaction.into(),
                Signed::PostEip155Legacy(transaction) => transaction.into(),
                Signed::Eip2930(transaction) => transaction.into(),
                Signed::Eip1559(transaction) => transaction.into(),
                Signed::Eip4844(transaction) => transaction.into(),
            }
        }
    }
}
