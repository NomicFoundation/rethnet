use edr_eth::{BlockSpec, BlockTag, Eip1898BlockSpec, PreEip1898BlockSpec, B256};

use super::key::CacheKeyVariant;

/// A block argument specification that is potentially cacheable.
#[derive(Clone, Debug)]
pub enum CacheableBlockSpec<'a> {
    /// Block number
    Number {
        /// Block number
        block_number: u64,
    },
    /// Block hash
    Hash {
        /// Block hash
        block_hash: &'a B256,
        /// Whether an error should be returned if the block is not canonical
        require_canonical: Option<bool>,
    },
    /// "earliest" block tag
    Earliest,
    /// "safe" block tag
    Safe,
    /// "finalized" block tag
    Finalized,
}

impl CacheKeyVariant for CacheableBlockSpec<'_> {
    fn cache_key_variant(&self) -> u8 {
        match self {
            CacheableBlockSpec::Number { .. } => 0,
            CacheableBlockSpec::Hash { .. } => 1,
            CacheableBlockSpec::Earliest => 2,
            CacheableBlockSpec::Safe => 3,
            CacheableBlockSpec::Finalized => 4,
        }
    }
}

/// Error type for [`CacheableBlockSpec::try_from`].
#[derive(thiserror::Error, Debug)]
#[error("Block spec is not cacheable: {0:?}")]
pub struct BlockSpecNotCacheableError(Option<BlockSpec>);

impl<'a> TryFrom<&'a BlockSpec> for CacheableBlockSpec<'a> {
    type Error = BlockSpecNotCacheableError;

    fn try_from(value: &'a BlockSpec) -> Result<Self, Self::Error> {
        match value {
            BlockSpec::Number(block_number) => Ok(CacheableBlockSpec::Number {
                block_number: *block_number,
            }),
            BlockSpec::Tag(tag) => match tag {
                // Latest and pending can be never resolved to a safe block number.
                BlockTag::Latest | BlockTag::Pending => {
                    Err(BlockSpecNotCacheableError(Some(value.clone())))
                }
                // Earliest, safe and finalized are potentially resolvable to a safe block number.
                BlockTag::Earliest => Ok(CacheableBlockSpec::Earliest),
                BlockTag::Safe => Ok(CacheableBlockSpec::Safe),
                BlockTag::Finalized => Ok(CacheableBlockSpec::Finalized),
            },
            BlockSpec::Eip1898(spec) => match spec {
                Eip1898BlockSpec::Hash {
                    block_hash,
                    require_canonical,
                } => Ok(CacheableBlockSpec::Hash {
                    block_hash,
                    require_canonical: *require_canonical,
                }),
                Eip1898BlockSpec::Number { block_number } => Ok(CacheableBlockSpec::Number {
                    block_number: *block_number,
                }),
            },
        }
    }
}

impl<'a> TryFrom<&'a Option<BlockSpec>> for CacheableBlockSpec<'a> {
    type Error = BlockSpecNotCacheableError;

    fn try_from(value: &'a Option<BlockSpec>) -> Result<Self, Self::Error> {
        match value {
            None => Err(BlockSpecNotCacheableError(None)),
            Some(block_spec) => CacheableBlockSpec::try_from(block_spec),
        }
    }
}

/// Error type for [`CacheableBlockSpec::try_from`].
#[derive(thiserror::Error, Debug)]
#[error("Block spec is not cacheable: {0:?}")]
pub struct PreEip1898BlockSpecNotCacheableError(PreEip1898BlockSpec);

impl<'a> TryFrom<&'a PreEip1898BlockSpec> for CacheableBlockSpec<'a> {
    type Error = PreEip1898BlockSpecNotCacheableError;

    fn try_from(value: &'a PreEip1898BlockSpec) -> Result<Self, Self::Error> {
        match value {
            PreEip1898BlockSpec::Number(block_number) => Ok(CacheableBlockSpec::Number {
                block_number: *block_number,
            }),
            PreEip1898BlockSpec::Tag(tag) => match tag {
                // Latest and pending can never be resolved to a safe block number.
                BlockTag::Latest | BlockTag::Pending => {
                    Err(PreEip1898BlockSpecNotCacheableError(value.clone()))
                }
                // Earliest, safe and finalized are potentially resolvable to a safe block number.
                BlockTag::Earliest => Ok(CacheableBlockSpec::Earliest),
                BlockTag::Safe => Ok(CacheableBlockSpec::Safe),
                BlockTag::Finalized => Ok(CacheableBlockSpec::Finalized),
            },
        }
    }
}

/// Error type for [`crate::cache::KeyHasher::hash_block_spec`].
#[derive(thiserror::Error, Debug)]
#[error("A block tag is not resolved.")]
pub struct UnresolvedBlockTagError;
