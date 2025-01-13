use alloy_rlp::Encodable;
use edr_eth::{
    eips::{eip2930, eip7702::AuthorizationList},
    keccak256,
    transaction::{ExecutableTransaction, Transaction, TxKind},
    utils::enveloped,
    Address, Bytes, B256, U256,
};

use super::Deposit;

impl Deposit {
    /// The type identifier for a deposit transaction.
    pub const TYPE: u8 = 0x7E;
}

impl ExecutableTransaction for Deposit {
    fn effective_gas_price(&self, _block_base_fee: U256) -> Option<U256> {
        None
    }

    fn max_fee_per_gas(&self) -> Option<&U256> {
        Some(self.gas_price())
    }

    fn rlp_encoding(&self) -> &Bytes {
        self.rlp_encoding.get_or_init(|| {
            let mut encoded = Vec::with_capacity(1 + self.length());
            enveloped(Self::TYPE, self, &mut encoded);
            encoded.into()
        })
    }

    fn total_blob_gas(&self) -> Option<u64> {
        None
    }

    fn transaction_hash(&self) -> &B256 {
        self.hash.get_or_init(|| keccak256(self.rlp_encoding()))
    }
}

impl PartialEq for Deposit {
    fn eq(&self, other: &Self) -> bool {
        // Custom implementation of `PartialEq` to ignore the `hash` field.
        self.source_hash == other.source_hash
            && self.from == other.from
            && self.to == other.to
            && self.mint == other.mint
            && self.value == other.value
            && self.gas_limit == other.gas_limit
            && self.is_system_tx == other.is_system_tx
            && self.data == other.data
    }
}

impl Transaction for Deposit {
    fn caller(&self) -> &Address {
        &self.from
    }

    fn gas_limit(&self) -> u64 {
        self.gas_limit
    }

    fn gas_price(&self) -> &U256 {
        // No gas is refunded as ETH. (either by not refunding or utilizing the fact the
        // gas-price of the deposit is 0)
        &U256::ZERO
    }

    fn kind(&self) -> TxKind {
        self.to
    }

    fn value(&self) -> &U256 {
        &self.value
    }

    fn data(&self) -> &Bytes {
        &self.data
    }

    fn nonce(&self) -> u64 {
        // Before Regolith: the nonce is always 0
        // With Regolith: the nonce is set to the depositNonce attribute of the
        // corresponding transaction receipt.
        0
    }

    fn chain_id(&self) -> Option<u64> {
        None
    }

    fn access_list(&self) -> &[eip2930::AccessListItem] {
        &[]
    }

    fn max_priority_fee_per_gas(&self) -> Option<&U256> {
        // No transaction priority fee is charged. No payment is made to the block
        // fee-recipient.
        Some(&U256::ZERO)
    }

    fn blob_hashes(&self) -> &[B256] {
        &[]
    }

    fn max_fee_per_blob_gas(&self) -> Option<&U256> {
        None
    }

    fn authorization_list(&self) -> Option<&AuthorizationList> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{str::FromStr as _, sync::OnceLock};

    use edr_eth::{
        address, b256,
        transaction::{ExecutableTransaction as _, TxKind},
        Bytes, U256,
    };

    use super::*;
    use crate::transaction;

    #[test]
    fn transaction_hash() -> anyhow::Result<()> {
        // Source:
        // <https://optimism.blockscout.com/tx/0xcca2f31992022e3a833959c505de021285a7c5339c8d1b8ad75100074e1c6aea>
        const EXPECTED: B256 =
            b256!("cca2f31992022e3a833959c505de021285a7c5339c8d1b8ad75100074e1c6aea");

        let transaction = transaction::Signed::Deposit(Deposit {
            source_hash: b256!("8672083ef2a54fb901eab5c1366a77c1e2c421793467cf1ea7925f21282804bb"),
            from: address!("deaddeaddeaddeaddeaddeaddeaddeaddead0001"),
            to: TxKind::Call(address!("4200000000000000000000000000000000000015")),
            mint: 0,
            value: U256::ZERO,
            gas_limit: 1000000,
            is_system_tx: false,
            data: Bytes::from_str("440a5e2000000558000c5fc5000000000000000500000000667b267b000000000133c922000000000000000000000000000000000000000000000000000000017a2aaed000000000000000000000000000000000000000000000000000000000000000015b91c0cfdec539cc0ad4dadc77c6dd693f5116721d5cc39b73b85aefe7501b480000000000000000000000006887246668a3b87f54deb3b94ba47a6f63f32985")?,
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        });

        assert_eq!(*transaction.transaction_hash(), EXPECTED);

        Ok(())
    }
}
