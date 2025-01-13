mod eip4844;

pub use self::eip4844::Eip4844;
use crate::{
    eips::{eip2930, eip7702},
    transaction::{
        signed::PreOrPostEip155, ExecutableTransaction, IsEip155, Signed, Transaction, TxKind,
        INVALID_TX_TYPE_ERROR_MESSAGE,
    },
    utils::enveloped,
    Address, Bytes, B256, U256,
};

pub type Legacy = super::signed::Legacy;
pub type Eip155 = super::signed::Eip155;
pub type Eip2930 = super::signed::Eip2930;
pub type Eip1559 = super::signed::Eip1559;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PooledTransaction {
    /// Legacy transaction
    PreEip155Legacy(Legacy),
    /// EIP-155 transaction
    PostEip155Legacy(Eip155),
    /// EIP-2930 transaction
    Eip2930(Eip2930),
    /// EIP-1559 transaction
    Eip1559(Eip1559),
    /// EIP-4844 transaction
    Eip4844(Eip4844),
}

impl PooledTransaction {
    /// Returns the blobs of the EIP-4844 transaction, if any.
    pub fn blobs(&self) -> Option<&[c_kzg::Blob]> {
        match self {
            PooledTransaction::Eip4844(tx) => Some(tx.blobs()),
            _ => None,
        }
    }

    /// Returns the commitments of the EIP-4844 transaction, if any.
    pub fn commitments(&self) -> Option<&[c_kzg::Bytes48]> {
        match self {
            PooledTransaction::Eip4844(tx) => Some(tx.commitments()),
            _ => None,
        }
    }

    /// Converts the pooled transaction into a signed transaction.
    pub fn into_payload(self) -> Signed {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => Signed::PreEip155Legacy(tx),
            PooledTransaction::PostEip155Legacy(tx) => Signed::PostEip155Legacy(tx),
            PooledTransaction::Eip2930(tx) => Signed::Eip2930(tx),
            PooledTransaction::Eip1559(tx) => Signed::Eip1559(tx),
            PooledTransaction::Eip4844(tx) => Signed::Eip4844(tx.into_payload()),
        }
    }

    /// Returns the proofs of the EIP-4844 transaction, if any.
    pub fn proofs(&self) -> Option<&[c_kzg::Bytes48]> {
        match self {
            PooledTransaction::Eip4844(tx) => Some(tx.proofs()),
            _ => None,
        }
    }
}

impl alloy_rlp::Decodable for PooledTransaction {
    fn decode(buf: &mut &[u8]) -> alloy_rlp::Result<Self> {
        use alloy_rlp::Buf;

        fn is_list(byte: u8) -> bool {
            byte >= 0xc0
        }

        let first = buf.first().ok_or(alloy_rlp::Error::InputTooShort)?;

        match *first {
            Eip2930::TYPE => {
                buf.advance(1);

                Ok(PooledTransaction::Eip2930(Eip2930::decode(buf)?))
            }
            Eip1559::TYPE => {
                buf.advance(1);

                Ok(PooledTransaction::Eip1559(Eip1559::decode(buf)?))
            }
            Eip4844::TYPE => {
                buf.advance(1);

                Ok(PooledTransaction::Eip4844(Eip4844::decode(buf)?))
            }
            byte if is_list(byte) => {
                let transaction = PreOrPostEip155::decode(buf)?;
                Ok(transaction.into())
            }
            _ => Err(alloy_rlp::Error::Custom(INVALID_TX_TYPE_ERROR_MESSAGE)),
        }
    }
}

impl alloy_rlp::Encodable for PooledTransaction {
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.encode(out),
            PooledTransaction::PostEip155Legacy(tx) => tx.encode(out),
            PooledTransaction::Eip2930(tx) => enveloped(1, tx, out),
            PooledTransaction::Eip1559(tx) => enveloped(2, tx, out),
            PooledTransaction::Eip4844(tx) => enveloped(3, tx, out),
        }
    }

    fn length(&self) -> usize {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.length(),
            PooledTransaction::PostEip155Legacy(tx) => tx.length(),
            PooledTransaction::Eip2930(tx) => tx.length() + 1,
            PooledTransaction::Eip1559(tx) => tx.length() + 1,
            PooledTransaction::Eip4844(tx) => tx.length() + 1,
        }
    }
}

impl ExecutableTransaction for PooledTransaction {
    fn effective_gas_price(&self, block_base_fee: U256) -> Option<U256> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            PooledTransaction::PostEip155Legacy(tx) => tx.effective_gas_price(block_base_fee),
            PooledTransaction::Eip2930(tx) => tx.effective_gas_price(block_base_fee),
            PooledTransaction::Eip1559(tx) => tx.effective_gas_price(block_base_fee),
            PooledTransaction::Eip4844(tx) => tx.effective_gas_price(block_base_fee),
        }
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.max_fee_per_gas(),
            PooledTransaction::PostEip155Legacy(tx) => tx.max_fee_per_gas(),
            PooledTransaction::Eip2930(tx) => tx.max_fee_per_gas(),
            PooledTransaction::Eip1559(tx) => tx.max_fee_per_gas(),
            PooledTransaction::Eip4844(tx) => tx.max_fee_per_gas(),
        }
    }

    fn rlp_encoding(&self) -> &Bytes {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.rlp_encoding(),
            PooledTransaction::PostEip155Legacy(tx) => tx.rlp_encoding(),
            PooledTransaction::Eip2930(tx) => tx.rlp_encoding(),
            PooledTransaction::Eip1559(tx) => tx.rlp_encoding(),
            PooledTransaction::Eip4844(tx) => tx.rlp_encoding(),
        }
    }

    fn total_blob_gas(&self) -> Option<u64> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.total_blob_gas(),
            PooledTransaction::PostEip155Legacy(tx) => tx.total_blob_gas(),
            PooledTransaction::Eip2930(tx) => tx.total_blob_gas(),
            PooledTransaction::Eip1559(tx) => tx.total_blob_gas(),
            PooledTransaction::Eip4844(tx) => tx.total_blob_gas(),
        }
    }

    fn transaction_hash(&self) -> &B256 {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.transaction_hash(),
            PooledTransaction::PostEip155Legacy(tx) => tx.transaction_hash(),
            PooledTransaction::Eip2930(tx) => tx.transaction_hash(),
            PooledTransaction::Eip1559(tx) => tx.transaction_hash(),
            PooledTransaction::Eip4844(tx) => tx.transaction_hash(),
        }
    }
}

impl From<Legacy> for PooledTransaction {
    fn from(value: Legacy) -> Self {
        PooledTransaction::PreEip155Legacy(value)
    }
}

impl From<Eip155> for PooledTransaction {
    fn from(value: Eip155) -> Self {
        PooledTransaction::PostEip155Legacy(value)
    }
}

impl From<Eip2930> for PooledTransaction {
    fn from(value: Eip2930) -> Self {
        PooledTransaction::Eip2930(value)
    }
}

impl From<Eip1559> for PooledTransaction {
    fn from(value: Eip1559) -> Self {
        PooledTransaction::Eip1559(value)
    }
}

impl From<Eip4844> for PooledTransaction {
    fn from(value: Eip4844) -> Self {
        PooledTransaction::Eip4844(value)
    }
}

impl From<PreOrPostEip155> for PooledTransaction {
    fn from(value: PreOrPostEip155) -> Self {
        match value {
            PreOrPostEip155::Pre(tx) => Self::PreEip155Legacy(tx),
            PreOrPostEip155::Post(tx) => Self::PostEip155Legacy(tx),
        }
    }
}

impl From<PooledTransaction> for Signed {
    fn from(value: PooledTransaction) -> Self {
        value.into_payload()
    }
}

impl IsEip155 for PooledTransaction {
    fn is_eip155(&self) -> bool {
        matches!(self, PooledTransaction::PostEip155Legacy(_))
    }
}

impl Transaction for PooledTransaction {
    fn caller(&self) -> &Address {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.caller(),
            PooledTransaction::PostEip155Legacy(tx) => tx.caller(),
            PooledTransaction::Eip2930(tx) => tx.caller(),
            PooledTransaction::Eip1559(tx) => tx.caller(),
            PooledTransaction::Eip4844(tx) => tx.caller(),
        }
    }

    fn gas_limit(&self) -> u64 {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.gas_limit(),
            PooledTransaction::PostEip155Legacy(tx) => tx.gas_limit(),
            PooledTransaction::Eip2930(tx) => tx.gas_limit(),
            PooledTransaction::Eip1559(tx) => tx.gas_limit(),
            PooledTransaction::Eip4844(tx) => tx.gas_limit(),
        }
    }

    fn gas_price(&self) -> &U256 {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.gas_price(),
            PooledTransaction::PostEip155Legacy(tx) => tx.gas_price(),
            PooledTransaction::Eip2930(tx) => tx.gas_price(),
            PooledTransaction::Eip1559(tx) => tx.gas_price(),
            PooledTransaction::Eip4844(tx) => tx.gas_price(),
        }
    }

    fn kind(&self) -> TxKind {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.kind(),
            PooledTransaction::PostEip155Legacy(tx) => tx.kind(),
            PooledTransaction::Eip2930(tx) => tx.kind(),
            PooledTransaction::Eip1559(tx) => tx.kind(),
            PooledTransaction::Eip4844(tx) => tx.kind(),
        }
    }

    fn value(&self) -> &U256 {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.value(),
            PooledTransaction::PostEip155Legacy(tx) => tx.value(),
            PooledTransaction::Eip2930(tx) => tx.value(),
            PooledTransaction::Eip1559(tx) => tx.value(),
            PooledTransaction::Eip4844(tx) => tx.value(),
        }
    }

    fn data(&self) -> &Bytes {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.data(),
            PooledTransaction::PostEip155Legacy(tx) => tx.data(),
            PooledTransaction::Eip2930(tx) => tx.data(),
            PooledTransaction::Eip1559(tx) => tx.data(),
            PooledTransaction::Eip4844(tx) => tx.data(),
        }
    }

    fn nonce(&self) -> u64 {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.nonce(),
            PooledTransaction::PostEip155Legacy(tx) => tx.nonce(),
            PooledTransaction::Eip2930(tx) => tx.nonce(),
            PooledTransaction::Eip1559(tx) => tx.nonce(),
            PooledTransaction::Eip4844(tx) => tx.nonce(),
        }
    }

    fn chain_id(&self) -> Option<u64> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.chain_id(),
            PooledTransaction::PostEip155Legacy(tx) => tx.chain_id(),
            PooledTransaction::Eip2930(tx) => tx.chain_id(),
            PooledTransaction::Eip1559(tx) => tx.chain_id(),
            PooledTransaction::Eip4844(tx) => tx.chain_id(),
        }
    }

    fn access_list(&self) -> &[eip2930::AccessListItem] {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.access_list(),
            PooledTransaction::PostEip155Legacy(tx) => tx.access_list(),
            PooledTransaction::Eip2930(tx) => tx.access_list(),
            PooledTransaction::Eip1559(tx) => tx.access_list(),
            PooledTransaction::Eip4844(tx) => tx.access_list(),
        }
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            PooledTransaction::PostEip155Legacy(tx) => tx.max_priority_fee_per_gas(),
            PooledTransaction::Eip2930(tx) => tx.max_priority_fee_per_gas(),
            PooledTransaction::Eip1559(tx) => tx.max_priority_fee_per_gas(),
            PooledTransaction::Eip4844(tx) => tx.max_priority_fee_per_gas(),
        }
    }

    fn blob_hashes(&self) -> &[B256] {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.blob_hashes(),
            PooledTransaction::PostEip155Legacy(tx) => tx.blob_hashes(),
            PooledTransaction::Eip2930(tx) => tx.blob_hashes(),
            PooledTransaction::Eip1559(tx) => tx.blob_hashes(),
            PooledTransaction::Eip4844(tx) => tx.blob_hashes(),
        }
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            PooledTransaction::PostEip155Legacy(tx) => tx.max_fee_per_blob_gas(),
            PooledTransaction::Eip2930(tx) => tx.max_fee_per_blob_gas(),
            PooledTransaction::Eip1559(tx) => tx.max_fee_per_blob_gas(),
            PooledTransaction::Eip4844(tx) => tx.max_fee_per_blob_gas(),
        }
    }

    fn authorization_list(&self) -> Option<&eip7702::AuthorizationList> {
        match self {
            PooledTransaction::PreEip155Legacy(tx) => tx.authorization_list(),
            PooledTransaction::PostEip155Legacy(tx) => tx.authorization_list(),
            PooledTransaction::Eip2930(tx) => tx.authorization_list(),
            PooledTransaction::Eip1559(tx) => tx.authorization_list(),
            PooledTransaction::Eip4844(tx) => tx.authorization_list(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{str::FromStr, sync::OnceLock};

    use alloy_rlp::Decodable;
    use c_kzg::BYTES_PER_BLOB;

    use super::*;
    use crate::{
        eips::eip4844::EnvKzgSettings,
        signature,
        transaction::{self, TxKind},
        Address, Bytes, B256, U256,
    };

    fn fake_eip4844_blob() -> c_kzg::Blob {
        const BLOB_VALUE: &[u8] = b"hello world";

        // The blob starts 0, followed by `hello world`, then 0x80, and is padded with
        // zeroes.
        let mut bytes = vec![0x0u8];
        bytes.append(&mut BLOB_VALUE.to_vec());
        bytes.push(alloy_rlp::EMPTY_STRING_CODE);

        bytes.resize(BYTES_PER_BLOB, 0);

        c_kzg::Blob::from_bytes(bytes.as_slice()).expect("Invalid blob")
    }

    macro_rules! impl_test_pooled_transaction_encoding_round_trip {
        ($(
            $name:ident => $transaction:expr,
        )+) => {
            $(
                paste::item! {
                    #[test]
                    fn [<pooled_transaction_encoding_round_trip_ $name>]() -> anyhow::Result<()> {
                        let transaction = $transaction;

                        let encoded = alloy_rlp::encode(&transaction);
                        let decoded = PooledTransaction::decode(&mut encoded.as_slice()).unwrap();

                        assert_eq!(decoded, transaction);

                        Ok(())
                    }
                }
            )+
        };
    }

    impl_test_pooled_transaction_encoding_round_trip! {
        pre_eip155 => PooledTransaction::PreEip155Legacy(Legacy {
            nonce: 0,
            gas_price: U256::from(1),
            gas_limit: 2,
            kind: TxKind::Call(Address::default()),
            value: U256::from(3),
            input: Bytes::from(vec![1, 2]),
            // SAFETY: Signature and caller address have been precomputed based on
            // `crate::edr_eth::transaction::signed::impl_test_signed_transaction_encoding_round_trip!`
            signature: unsafe { signature::Fakeable::with_address_unchecked(
                signature::SignatureWithRecoveryId {
                    r: U256::from_str("0xf0407adecc60467f3293582a9e1d726db5bc6b64f230bfb6ff04f23a1bbfe8dc")?,
                    s: U256::from_str("0x2f68623b42c3b302b8b96035c30ca58c566fdfdc3421ddb4f41d61b485e1401b")?,
                    v: 27,
                },
                Address::from_str("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266")?,
            )},
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        }),
        post_eip155 => PooledTransaction::PostEip155Legacy(Eip155 {
            nonce: 0,
            gas_price: U256::from(1),
            gas_limit: 2,
            kind: TxKind::Create,
            value: U256::from(3),
            input: Bytes::from(vec![1, 2]),
            // SAFETY: Signature and caller address have been precomputed based on
            // `crate::edr_eth::transaction::signed::impl_test_signed_transaction_encoding_round_trip!`
            signature: unsafe { signature::Fakeable::with_address_unchecked(
                signature::SignatureWithRecoveryId {
                    r: U256::from_str("0xed3a859fce13d142bba6051a91f934947f71c5f8ce8e3fe5bc7a845365309b90")?,
                    s: U256::from_str("0x1deb3bbf3fff7fba96e853ff1a19eabb117ef93b7704176893e4e9fff0e04576")?,
                    v: 2709,
                },
                Address::from_str("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266")?,
            )},
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        }),
        eip2930 => PooledTransaction::Eip2930(Eip2930 {
            chain_id: 1,
            nonce: 0,
            gas_price: U256::from(1),
            gas_limit: 2,
            kind: TxKind::Call(Address::default()),
            value: U256::from(3),
            input: Bytes::from(vec![1, 2]),
            access_list: vec![].into(),
            // SAFETY: Signature and caller address have been precomputed based on
            // `crate::edr_eth::transaction::signed::impl_test_signed_transaction_encoding_round_trip!`
            signature: unsafe { signature::Fakeable::with_address_unchecked(
                signature::SignatureWithYParity {
                    r: U256::from_str("0xa8d41ec812e66a7d80a1478f053cb8b627abb36191f53c2f7a153b4e4f90564d")?,
                    s: U256::from_str("0x5a04b306c280730f872be5cd1970a3b493bdb4855fdbc2725dd9452f1a3e9412")?,
                    y_parity: true,
                },
                Address::from_str("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266")?,
            )},
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        }),
        eip1559 => PooledTransaction::Eip1559(Eip1559 {
            chain_id: 1,
            nonce: 0,
            max_priority_fee_per_gas: U256::from(1),
            max_fee_per_gas: U256::from(2),
            gas_limit: 3,
            kind: TxKind::Create,
            value: U256::from(4),
            input: Bytes::from(vec![1, 2]),
            access_list: vec![].into(),
            // SAFETY: Signature and caller address have been precomputed based on
            // `crate::edr_eth::transaction::signed::impl_test_signed_transaction_encoding_round_trip!`
            signature: unsafe { signature::Fakeable::with_address_unchecked(
                signature::SignatureWithYParity {
                    r: U256::from_str("0x263b71578125bf86e9e842a920af2d941cd023893c4a452d158c87eabdf06bb9")?,
                    s: U256::from_str("0x097ff1980e38856c8e0310823e6cfc83032314f50ddd38568d3c9cf93e47d517")?,
                    y_parity: true,
                },
                Address::from_str("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266")?,
            )},
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        }),
        eip4844 => PooledTransaction::Eip4844(
            Eip4844::new(transaction::signed::Eip4844 {
                chain_id: 1337,
                nonce: 0,
                max_priority_fee_per_gas: U256::from(1_000_000_000),
                max_fee_per_gas: U256::from(1_000_000_000),
                max_fee_per_blob_gas: U256::from(1),
                gas_limit: 1_000_000,
                to: Address::ZERO,
                value: U256::ZERO,
                input: Bytes::from_str("0x2069b0c7")?,
                access_list: vec![].into(),
                blob_hashes: vec![B256::from_str("0x01ae39c06daecb6a178655e3fab2e56bd61e81392027947529e4def3280c546e")?],
                // SAFETY: Signature and caller address have been precomputed
                signature: unsafe { signature::Fakeable::with_address_unchecked(
                    signature::SignatureWithYParity {
                        r: U256::from_str("0xaeb099417be87077fe470104f6aa73e4e473a51a6c4be62607d10e8f13f9d082")?,
                        s: U256::from_str("0x390a4c98aaecf0cfc2b27e68bdcec511dd4136356197e5937ce186af5608690b")?,
                        y_parity: true,
                    },
                    Address::from_str("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266")?,
                )},
                hash: OnceLock::new(),
                rlp_encoding: OnceLock::new(),
            },
            vec![fake_eip4844_blob()],
            vec![c_kzg::Bytes48::from_hex(
                "b93ab7583ad8a57b2edd262889391f37a83ab41107dc02c1a68220841379ae828343e84ac1c70fb7c2640ee3522c4c36"
            ).expect("Invalid commitment")],
            vec![c_kzg::Bytes48::from_hex(
                "86ffb073648261475af77cc902c5189bf3d33d0f63e025f23c69ac1e4cc0a7646e1a59ff8e5600f0fcc35f78fe1a4df2"
            ).expect("Invalid proof")], EnvKzgSettings::Default.get())?
        ),
    }
}
