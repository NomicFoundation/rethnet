use std::{num::NonZeroU64, path::PathBuf, time::SystemTime};

use edr_eth::{
    account::AccountInfo, block::BlobGas, spec::HardforkTrait, Address, ChainId, HashMap, B256,
    U256,
};
use edr_evm::{hardfork, state::EvmStorage, MineOrdering};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::requests::{hardhat::rpc_types::ForkConfig, IntervalConfig as IntervalConfigRequest};

/// Configuration of an account and its storage.
///
/// Similar to `edr_eth::Account` but without the `status` field.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Account {
    /// Balance, nonce, and code.
    pub info: AccountInfo,
    /// Storage cache
    pub storage: EvmStorage,
}

impl From<AccountInfo> for Account {
    fn from(info: AccountInfo) -> Self {
        Self {
            info,
            storage: HashMap::new(),
        }
    }
}

/// Configuration for interval mining.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Interval {
    Fixed(NonZeroU64),
    Range { min: u64, max: u64 },
}

impl Interval {
    /// Generates a (random) interval based on the configuration.
    pub fn generate_interval(&self) -> u64 {
        match self {
            Interval::Fixed(interval) => interval.get(),
            Interval::Range { min, max } => rand::thread_rng().gen_range(*min..=*max),
        }
    }
}

/// An error that occurs when trying to convert [`IntervalConfigRequest`] to an
/// `Option<IntervalConfig>`.
#[derive(Debug, thiserror::Error)]
pub enum IntervalConfigConversionError {
    /// The minimum value in the range is greater than the maximum value.
    #[error("Minimum value in range is greater than maximum value")]
    MinGreaterThanMax,
}

impl TryInto<Option<Interval>> for IntervalConfigRequest {
    type Error = IntervalConfigConversionError;

    fn try_into(self) -> Result<Option<Interval>, Self::Error> {
        match self {
            Self::FixedOrDisabled(0) => Ok(None),
            Self::FixedOrDisabled(value) => {
                // Zero implies disabled
                Ok(NonZeroU64::new(value).map(Interval::Fixed))
            }
            Self::Range([min, max]) => {
                if max >= min {
                    Ok(Some(Interval::Range { min, max }))
                } else {
                    Err(IntervalConfigConversionError::MinGreaterThanMax)
                }
            }
        }
    }
}

/// Configuration for the provider's mempool.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemPool {
    pub order: MineOrdering,
}

/// Configuration for the provider's miner.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Mining {
    pub auto_mine: bool,
    pub interval: Option<Interval>,
    pub mem_pool: MemPool,
}

/// Configuration for the provider
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Provider<HardforkT: HardforkTrait> {
    pub allow_blocks_with_same_timestamp: bool,
    pub allow_unlimited_contract_size: bool,
    pub accounts: Vec<OwnedAccount>,
    /// Whether to return an `Err` when `eth_call` fails
    pub bail_on_call_failure: bool,
    /// Whether to return an `Err` when a `eth_sendTransaction` fails
    pub bail_on_transaction_failure: bool,
    pub block_gas_limit: NonZeroU64,
    pub cache_dir: PathBuf,
    pub chain_id: ChainId,
    pub chains: HashMap<ChainId, hardfork::Activations<HardforkT>>,
    pub coinbase: Address,
    pub enable_rip_7212: bool,
    pub fork: Option<ForkConfig>,
    pub genesis_state: HashMap<Address, Account>,
    pub hardfork: HardforkT,
    pub initial_base_fee_per_gas: Option<U256>,
    pub initial_blob_gas: Option<BlobGas>,
    pub initial_date: Option<SystemTime>,
    pub initial_parent_beacon_block_root: Option<B256>,
    pub min_gas_price: U256,
    pub mining: Mining,
    pub network_id: u64,
}

/// Configuration input for a single account
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OwnedAccount {
    /// the secret key of the account
    #[serde(with = "secret_key_serde")]
    pub secret_key: k256::SecretKey,
    /// the balance of the account
    pub balance: U256,
}

mod secret_key_serde {
    use edr_eth::signature::{secret_key_from_str, secret_key_to_str};
    use serde::Deserialize;

    pub(super) fn serialize<S>(
        secret_key: &k256::SecretKey,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&secret_key_to_str(secret_key))
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<k256::SecretKey, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = <&str as Deserialize>::deserialize(deserializer)?;
        secret_key_from_str(s).map_err(serde::de::Error::custom)
    }
}

impl Default for MemPool {
    fn default() -> Self {
        Self {
            order: MineOrdering::Priority,
        }
    }
}

impl Default for Mining {
    fn default() -> Self {
        Self {
            auto_mine: true,
            interval: None,
            mem_pool: MemPool::default(),
        }
    }
}
