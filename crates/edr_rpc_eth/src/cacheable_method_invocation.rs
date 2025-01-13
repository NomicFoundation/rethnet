use edr_eth::{reward_percentile::RewardPercentile, Address, B256, U256};
use edr_rpc_client::{
    cache::{
        self,
        block_spec::{
            BlockSpecNotCacheableError, CacheableBlockSpec, PreEip1898BlockSpecNotCacheableError,
            UnresolvedBlockTagError,
        },
        filter::{CacheableLogFilterOptions, LogFilterOptionsNotCacheableError},
        key::{CacheKeyVariant, ReadCacheKey, WriteCacheKey},
        CacheableMethod,
    },
    RpcMethod,
};

use crate::request_methods::RequestMethod;

/// Potentially cacheable Ethereum JSON-RPC methods.
#[derive(Clone, Debug)]
pub enum CachedRequestMethod<'a> {
    /// `eth_feeHistory`
    FeeHistory {
        block_count: &'a U256,
        newest_block: CacheableBlockSpec<'a>,
        reward_percentiles: &'a Option<Vec<RewardPercentile>>,
    },
    /// `eth_getBalance`
    GetBalance {
        address: &'a Address,
        block_spec: CacheableBlockSpec<'a>,
    },
    /// `eth_getBlockByNumber`
    GetBlockByNumber {
        block_spec: CacheableBlockSpec<'a>,

        /// include transaction data
        include_tx_data: bool,
    },
    /// `eth_getBlockByHash`
    GetBlockByHash {
        /// hash
        block_hash: &'a B256,
        /// include transaction data
        include_tx_data: bool,
    },
    /// `eth_getCode`
    GetCode {
        address: &'a Address,
        block_spec: CacheableBlockSpec<'a>,
    },
    /// `eth_getLogs`
    GetLogs {
        params: CacheableLogFilterOptions<'a>,
    },
    /// `eth_getStorageAt`
    GetStorageAt {
        address: &'a Address,
        position: &'a U256,
        block_spec: CacheableBlockSpec<'a>,
    },
    /// `eth_getTransactionByHash`
    GetTransactionByHash { transaction_hash: &'a B256 },
    /// `eth_getTransactionCount`
    GetTransactionCount {
        address: &'a Address,
        block_spec: CacheableBlockSpec<'a>,
    },
    /// `eth_getTransactionReceipt`
    GetTransactionReceipt { transaction_hash: &'a B256 },
    /// `net_version`
    NetVersion,
}

impl CachedRequestMethod<'_> {
    // Allow to keep same structure as other RequestMethod and other methods.
    #[allow(clippy::match_same_arms)]
    fn key_hasher(&self) -> Result<cache::KeyHasher, UnresolvedBlockTagError> {
        let hasher = cache::KeyHasher::default();
        let hasher = hasher.hash_u8(self.cache_key_variant());

        let hasher = match self {
            CachedRequestMethod::FeeHistory {
                block_count,
                newest_block,
                reward_percentiles,
            } => {
                let hasher = hasher
                    .hash_u256(block_count)
                    .hash_block_spec(newest_block)?
                    .hash_u8(reward_percentiles.cache_key_variant());
                match reward_percentiles {
                    Some(reward_percentiles) => hasher.hash_reward_percentiles(reward_percentiles),
                    None => hasher,
                }
            }
            CachedRequestMethod::GetBalance {
                address,
                block_spec,
            } => hasher.hash_address(address).hash_block_spec(block_spec)?,
            CachedRequestMethod::GetBlockByNumber {
                block_spec,
                include_tx_data,
            } => hasher
                .hash_block_spec(block_spec)?
                .hash_bool(*include_tx_data),
            CachedRequestMethod::GetBlockByHash {
                block_hash,
                include_tx_data,
            } => hasher.hash_b256(block_hash).hash_bool(*include_tx_data),
            CachedRequestMethod::GetCode {
                address,
                block_spec,
            } => hasher.hash_address(address).hash_block_spec(block_spec)?,
            CachedRequestMethod::GetLogs { params } => hasher.hash_log_filter_options(params)?,
            CachedRequestMethod::GetStorageAt {
                address,
                position,
                block_spec,
            } => hasher
                .hash_address(address)
                .hash_u256(position)
                .hash_block_spec(block_spec)?,
            CachedRequestMethod::GetTransactionByHash { transaction_hash } => {
                hasher.hash_b256(transaction_hash)
            }
            CachedRequestMethod::GetTransactionCount {
                address,
                block_spec,
            } => hasher.hash_address(address).hash_block_spec(block_spec)?,
            CachedRequestMethod::GetTransactionReceipt { transaction_hash } => {
                hasher.hash_b256(transaction_hash)
            }
            CachedRequestMethod::NetVersion => hasher,
        };

        Ok(hasher)
    }
}

/// Error type for [`CacheableRequestMethod::try_from`].
#[derive(thiserror::Error, Debug)]
pub enum MethodNotCacheableError {
    #[error(transparent)]
    BlockSpec(#[from] BlockSpecNotCacheableError),
    #[error("Method is not cacheable: {0:?}")]
    RequestMethod(RequestMethod),
    #[error("Get logs input is not cacheable: {0:?}")]
    GetLogsInput(#[from] LogFilterOptionsNotCacheableError),
    #[error(transparent)]
    PreEip18989BlockSpec(#[from] PreEip1898BlockSpecNotCacheableError),
}

impl<'a> TryFrom<&'a RequestMethod> for CachedRequestMethod<'a> {
    type Error = MethodNotCacheableError;

    fn try_from(value: &'a RequestMethod) -> Result<Self, Self::Error> {
        match value {
            RequestMethod::FeeHistory(block_count, newest_block, reward_percentiles) => {
                Ok(CachedRequestMethod::FeeHistory {
                    block_count,
                    newest_block: newest_block.try_into()?,
                    reward_percentiles,
                })
            }
            RequestMethod::GetBalance(address, block_spec) => Ok(CachedRequestMethod::GetBalance {
                address,
                block_spec: block_spec.try_into()?,
            }),
            RequestMethod::GetBlockByNumber(block_spec, include_tx_data) => {
                Ok(CachedRequestMethod::GetBlockByNumber {
                    block_spec: block_spec.try_into()?,
                    include_tx_data: *include_tx_data,
                })
            }
            RequestMethod::GetBlockByHash(block_hash, include_tx_data) => {
                Ok(CachedRequestMethod::GetBlockByHash {
                    block_hash,
                    include_tx_data: *include_tx_data,
                })
            }
            RequestMethod::GetCode(address, block_spec) => Ok(CachedRequestMethod::GetCode {
                address,
                block_spec: block_spec.try_into()?,
            }),
            RequestMethod::GetLogs(params) => Ok(CachedRequestMethod::GetLogs {
                params: params.try_into()?,
            }),
            RequestMethod::GetStorageAt(address, position, block_spec) => {
                Ok(CachedRequestMethod::GetStorageAt {
                    address,
                    position,
                    block_spec: block_spec.try_into()?,
                })
            }
            RequestMethod::GetTransactionByHash(transaction_hash) => {
                Ok(CachedRequestMethod::GetTransactionByHash { transaction_hash })
            }
            RequestMethod::GetTransactionCount(address, block_spec) => {
                Ok(CachedRequestMethod::GetTransactionCount {
                    address,
                    block_spec: block_spec.try_into()?,
                })
            }
            RequestMethod::GetTransactionReceipt(transaction_hash) => {
                Ok(CachedRequestMethod::GetTransactionReceipt { transaction_hash })
            }
            RequestMethod::NetVersion(_) => Ok(CachedRequestMethod::NetVersion),

            // Explicit to make sure if a new method is added, it is not forgotten here.
            // Chain id is not cacheable since a remote might change its chain id e.g. if it's a
            // forked node running on localhost.
            RequestMethod::BlockNumber(_) | RequestMethod::ChainId(_) => {
                Err(MethodNotCacheableError::RequestMethod(value.clone()))
            }
        }
    }
}

/// Method invocations where, if the block spec argument is symbolic, it can be
/// resolved to a block number from the response.
#[derive(Debug, Clone)]
pub enum MethodWithResolvableBlockSpec {
    GetBlockByNumber { include_tx_data: bool },
}

impl<'method> From<CachedRequestMethod<'method>> for Option<MethodWithResolvableBlockSpec> {
    fn from(val: CachedRequestMethod<'method>) -> Self {
        match val {
            CachedRequestMethod::GetBlockByNumber {
                include_tx_data,
                block_spec: _,
            } => Some(MethodWithResolvableBlockSpec::GetBlockByNumber { include_tx_data }),
            _ => None,
        }
    }
}

impl RpcMethod for RequestMethod {
    type Cacheable<'method> = CachedRequestMethod<'method>;

    fn block_number_request() -> Self {
        Self::BlockNumber(())
    }

    fn chain_id_request() -> Self {
        Self::ChainId(())
    }

    #[cfg(feature = "tracing")]
    fn name(&self) -> &'static str {
        match self {
            Self::BlockNumber(_) => "eth_blockNumber",
            Self::FeeHistory(_, _, _) => "eth_feeHistory",
            Self::ChainId(_) => "eth_chainId",
            Self::GetBalance(_, _) => "eth_getBalance",
            Self::GetBlockByNumber(_, _) => "eth_getBlockByNumber",
            Self::GetBlockByHash(_, _) => "eth_getBlockByHash",
            Self::GetCode(_, _) => "eth_getCode",
            Self::GetLogs(_) => "eth_getLogs",
            Self::GetStorageAt(_, _, _) => "eth_getStorageAt",
            Self::GetTransactionByHash(_) => "eth_getTransactionByHash",
            Self::GetTransactionCount(_, _) => "eth_getTransactionCount",
            Self::GetTransactionReceipt(_) => "eth_getTransactionReceipt",
            Self::NetVersion(_) => "net_version",
        }
    }
}

impl CacheableMethod for CachedRequestMethod<'_> {
    type MethodWithResolvableBlockTag = MethodWithResolvableBlockSpec;

    fn resolve_block_tag(method: Self::MethodWithResolvableBlockTag, block_number: u64) -> Self {
        let resolved_block_spec = CacheableBlockSpec::Number { block_number };

        match method {
            MethodWithResolvableBlockSpec::GetBlockByNumber { include_tx_data } => {
                CachedRequestMethod::GetBlockByNumber {
                    block_spec: resolved_block_spec,
                    include_tx_data,
                }
            }
        }
    }

    fn read_cache_key(self) -> Option<ReadCacheKey> {
        let key_hasher = self.key_hasher().ok()?;
        Some(ReadCacheKey::finalize(key_hasher))
    }

    #[allow(clippy::match_same_arms)]
    fn write_cache_key(self) -> Option<WriteCacheKey<Self>> {
        match self.key_hasher() {
            Err(UnresolvedBlockTagError) => WriteCacheKey::needs_block_tag_resolution(self),
            Ok(hasher) => match self {
                CachedRequestMethod::FeeHistory {
                    block_count: _,
                    newest_block,
                    reward_percentiles: _,
                } => WriteCacheKey::needs_safety_check(hasher, newest_block),
                CachedRequestMethod::GetBalance {
                    address: _,
                    block_spec,
                } => WriteCacheKey::needs_safety_check(hasher, block_spec),
                CachedRequestMethod::GetBlockByNumber {
                    block_spec,
                    include_tx_data: _,
                } => WriteCacheKey::needs_safety_check(hasher, block_spec),
                CachedRequestMethod::GetBlockByHash {
                    block_hash: _,
                    include_tx_data: _,
                } => Some(WriteCacheKey::finalize(hasher)),
                CachedRequestMethod::GetCode {
                    address: _,
                    block_spec,
                } => WriteCacheKey::needs_safety_check(hasher, block_spec),
                CachedRequestMethod::GetLogs {
                    params: CacheableLogFilterOptions { range, .. },
                } => WriteCacheKey::needs_range_check(hasher, range),
                CachedRequestMethod::GetStorageAt {
                    address: _,
                    position: _,
                    block_spec,
                } => WriteCacheKey::needs_safety_check(hasher, block_spec),
                CachedRequestMethod::GetTransactionByHash {
                    transaction_hash: _,
                } => Some(WriteCacheKey::finalize(hasher)),
                CachedRequestMethod::GetTransactionCount {
                    address: _,
                    block_spec,
                } => WriteCacheKey::needs_safety_check(hasher, block_spec),
                CachedRequestMethod::GetTransactionReceipt {
                    transaction_hash: _,
                } => Some(WriteCacheKey::finalize(hasher)),
                CachedRequestMethod::NetVersion => Some(WriteCacheKey::finalize(hasher)),
            },
        }
    }
}

impl CacheKeyVariant for CachedRequestMethod<'_> {
    fn cache_key_variant(&self) -> u8 {
        match self {
            // The commented out methods have been removed as they're not currently in use by the
            // RPC client. If they're added back, they should keep their old variant
            // number. CacheableRequestMethod::ChainId => 0,
            CachedRequestMethod::GetBalance { .. } => 1,
            CachedRequestMethod::GetBlockByNumber { .. } => 2,
            CachedRequestMethod::GetBlockByHash { .. } => 3,
            // CacheableRequestMethod::GetBlockTransactionCountByHash { .. } => 4,
            // CacheableRequestMethod::GetBlockTransactionCountByNumber { .. } => 5,
            CachedRequestMethod::GetCode { .. } => 6,
            CachedRequestMethod::GetLogs { .. } => 7,
            CachedRequestMethod::GetStorageAt { .. } => 8,
            // CacheableRequestMethod::GetTransactionByBlockHashAndIndex { .. } => 9,
            // CacheableRequestMethod::GetTransactionByBlockNumberAndIndex { .. } => 10,
            CachedRequestMethod::GetTransactionByHash { .. } => 11,
            CachedRequestMethod::GetTransactionCount { .. } => 12,
            CachedRequestMethod::GetTransactionReceipt { .. } => 13,
            CachedRequestMethod::NetVersion => 14,
            CachedRequestMethod::FeeHistory { .. } => 15,
        }
    }
}

#[cfg(test)]
mod test {
    use edr_rpc_client::cache::filter::CacheableLogFilterRange;

    use super::*;

    #[test]
    fn test_hash_length() {
        let hash = cache::KeyHasher::default().hash_u8(0).finalize();
        // 32 bytes as hex
        assert_eq!(hash.len(), 2 * 32);
    }

    #[test]
    fn test_hasher_block_spec_hash_and_number_not_equal() {
        let block_number = u64::default();
        let block_hash = B256::default();

        let hash_one = cache::KeyHasher::default()
            .hash_block_spec(&CacheableBlockSpec::Hash {
                block_hash: &block_hash,
                require_canonical: None,
            })
            .unwrap()
            .finalize();
        let hash_two = cache::KeyHasher::default()
            .hash_block_spec(&CacheableBlockSpec::Number { block_number })
            .unwrap()
            .finalize();

        assert_ne!(hash_one, hash_two);
    }

    #[test]
    fn test_get_logs_input_from_to_matters() {
        let from = CacheableBlockSpec::Number { block_number: 1 };
        let to = CacheableBlockSpec::Number { block_number: 2 };
        let address = Address::default();

        let hash_one = cache::KeyHasher::default()
            .hash_log_filter_options(&CacheableLogFilterOptions {
                range: CacheableLogFilterRange::Range {
                    from_block: from.clone(),
                    to_block: to.clone(),
                },
                addresses: vec![&address],
                topics: Vec::new(),
            })
            .unwrap()
            .finalize();

        let hash_two = cache::KeyHasher::default()
            .hash_log_filter_options(&CacheableLogFilterOptions {
                range: CacheableLogFilterRange::Range {
                    from_block: to,
                    to_block: from,
                },
                addresses: vec![&address],
                topics: Vec::new(),
            })
            .unwrap()
            .finalize();

        assert_ne!(hash_one, hash_two);
    }

    #[test]
    fn test_same_arguments_keys_not_equal() {
        let value = B256::default();
        let key_one = CachedRequestMethod::GetTransactionByHash {
            transaction_hash: &value,
        }
        .read_cache_key()
        .unwrap();
        let key_two = CachedRequestMethod::GetTransactionReceipt {
            transaction_hash: &value,
        }
        .read_cache_key()
        .unwrap();

        assert_ne!(key_one, key_two);
    }

    #[test]
    fn test_get_storage_at_block_spec_is_taken_into_account() {
        let address = Address::default();
        let position = U256::default();

        let key_one = CachedRequestMethod::GetStorageAt {
            address: &address,
            position: &position,
            block_spec: CacheableBlockSpec::Hash {
                block_hash: &B256::default(),
                require_canonical: None,
            },
        }
        .read_cache_key()
        .unwrap();

        let key_two = CachedRequestMethod::GetStorageAt {
            address: &address,
            position: &position,
            block_spec: CacheableBlockSpec::Number {
                block_number: u64::default(),
            },
        }
        .read_cache_key()
        .unwrap();

        assert_ne!(key_one, key_two);
    }

    #[test]
    fn test_get_storage_at_block_same_matches() {
        let address = Address::default();
        let position = U256::default();
        let block_number = u64::default();
        let block_spec = CacheableBlockSpec::Number { block_number };

        let key_one = CachedRequestMethod::GetStorageAt {
            address: &address,
            position: &position,
            block_spec: block_spec.clone(),
        }
        .read_cache_key()
        .unwrap();

        let key_two = CachedRequestMethod::GetStorageAt {
            address: &address,
            position: &position,
            block_spec,
        }
        .read_cache_key()
        .unwrap();

        assert_eq!(key_one, key_two);
    }
}
