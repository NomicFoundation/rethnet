use std::sync::OnceLock;

use alloy_rlp::RlpEncodable;
use k256::SecretKey;

use crate::{
    eips::eip2930,
    keccak256,
    signature::{self, public_key_to_address, Fakeable, SignatureError},
    transaction,
    utils::envelop_bytes,
    Address, Bytes, B256, U256,
};

#[derive(Clone, Debug, PartialEq, Eq, RlpEncodable)]
pub struct Eip4844 {
    // The order of these fields determines encoding order.
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee_per_gas: U256,
    pub max_fee_per_gas: U256,
    pub gas_limit: u64,
    pub to: Address,
    pub value: U256,
    pub input: Bytes,
    pub access_list: Vec<eip2930::AccessListItem>,
    pub max_fee_per_blob_gas: U256,
    pub blob_hashes: Vec<B256>,
}

impl Eip4844 {
    /// The type identifier for an EIP-4844 transaction.
    pub const TYPE: u8 = 3;

    /// Computes the hash of the transaction.
    pub fn hash(&self) -> B256 {
        let encoded = alloy_rlp::encode(self);

        keccak256(envelop_bytes(3, &encoded))
    }

    /// Signs the transaction with the provided secret key.
    pub fn sign(
        self,
        secret_key: &SecretKey,
    ) -> Result<transaction::signed::Eip4844, SignatureError> {
        let caller = public_key_to_address(secret_key.public_key());

        // SAFETY: The caller is derived from the secret key.
        unsafe { self.sign_for_sender_unchecked(secret_key, caller) }
    }

    /// Signs the transaction with the provided secret key, belonging to the
    /// provided caller's address.
    ///
    /// # Safety
    ///
    /// The `caller` and `secret_key` must correspond to the same account.
    pub unsafe fn sign_for_sender_unchecked(
        self,
        secret_key: &SecretKey,
        caller: Address,
    ) -> Result<transaction::signed::Eip4844, SignatureError> {
        let hash = self.hash();
        let signature = signature::SignatureWithYParity::new(hash, secret_key)?;

        Ok(transaction::signed::Eip4844 {
            chain_id: self.chain_id,
            nonce: self.nonce,
            max_priority_fee_per_gas: self.max_priority_fee_per_gas,
            max_fee_per_gas: self.max_fee_per_gas,
            max_fee_per_blob_gas: self.max_fee_per_blob_gas,
            gas_limit: self.gas_limit,
            to: self.to,
            value: self.value,
            input: self.input,
            access_list: self.access_list.into(),
            blob_hashes: self.blob_hashes,
            signature: Fakeable::with_address_unchecked(signature, caller),
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        })
    }

    pub fn fake_sign(self, address: Address) -> transaction::signed::Eip4844 {
        transaction::signed::Eip4844 {
            chain_id: self.chain_id,
            nonce: self.nonce,
            max_priority_fee_per_gas: self.max_priority_fee_per_gas,
            max_fee_per_gas: self.max_fee_per_gas,
            max_fee_per_blob_gas: self.max_fee_per_blob_gas,
            gas_limit: self.gas_limit,
            to: self.to,
            value: self.value,
            input: self.input,
            access_list: self.access_list.into(),
            blob_hashes: self.blob_hashes,
            signature: signature::Fakeable::fake(address, None),
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::str::FromStr;

    use revm_primitives::b256;

    use super::*;
    use crate::transaction::fake_signature::tests::test_fake_sign_properties;

    fn dummy_request() -> Eip4844 {
        // From https://github.com/ethereumjs/ethereumjs-monorepo/blob/master/packages/tx/test/eip4844.spec.ts#L68
        Eip4844 {
            chain_id: 1337,
            nonce: 0,
            max_priority_fee_per_gas: U256::from(0x3b9aca00u64),
            max_fee_per_gas: U256::from(0x3b9aca00u64),
            gas_limit: 1000000u64,
            to: Address::ZERO,
            value: U256::ZERO,
            input: Bytes::from_str("0x2069b0c7").expect("Valid hex string"),
            access_list: Vec::new(),
            max_fee_per_blob_gas: U256::from(1u64),
            blob_hashes: vec![b256!(
                "01ae39c06daecb6a178655e3fab2e56bd61e81392027947529e4def3280c546e"
            )],
        }
    }

    test_fake_sign_properties!();

    // Hardhat doesn't support EIP-4844 yet, hence no fake signature test
    // vector.
    #[test]
    fn transaction_request_hash() {
        const EXPECTED: B256 =
            b256!("9dccf66bda0bd29f3a6fb35808360b041203b28c90236065cd4753cf97cfd5fd");

        let request = dummy_request();
        assert_eq!(*request.hash(), EXPECTED);
    }
}
