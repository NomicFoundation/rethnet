use edr_eth::signature::secret_key_from_str;
use napi::{
    bindgen_prelude::{BigInt, Uint8Array},
    Status,
};
use napi_derive::napi;

use crate::cast::TryCast as _;

/// A description of an account's state.
#[napi(object)]
pub struct Account {
    /// The account's address
    pub address: Uint8Array,
    /// The account's balance
    pub balance: BigInt,
    /// The account's nonce
    pub nonce: BigInt,
    /// The account's code
    pub code: Option<Uint8Array>,
    /// The account's storage
    pub storage: Vec<StorageSlot>,
}

/// A description of a storage slot's state.
#[napi(object)]
pub struct StorageSlot {
    /// The storage slot's index
    pub index: BigInt,
    /// The storage slot's value
    pub value: BigInt,
}

/// An owned account, for which the secret key is known, and its desired genesis
/// balance.
#[napi(object)]
pub struct OwnedAccount {
    /// Account secret key
    pub secret_key: String,
    /// Account balance
    pub balance: BigInt,
}

impl TryFrom<OwnedAccount> for edr_provider::config::OwnedAccount {
    type Error = napi::Error;

    fn try_from(value: OwnedAccount) -> Result<Self, Self::Error> {
        let secret_key = secret_key_from_str(&value.secret_key)
            .map_err(|e| napi::Error::new(Status::InvalidArg, e.to_string()))?;

        Ok(Self {
            secret_key,
            balance: value.balance.try_cast()?,
        })
    }
}
