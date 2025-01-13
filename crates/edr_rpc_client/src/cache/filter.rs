use edr_eth::{
    filter::{LogFilterOptions, OneOrMore},
    Address, B256,
};

use super::{block_spec::CacheableBlockSpec, key::CacheKeyVariant};

/// A cacheable input for the `eth_getLogs` method.
#[derive(Clone, Debug)]
pub struct CacheableLogFilterOptions<'a> {
    /// The range
    pub range: CacheableLogFilterRange<'a>,
    /// The addresses
    pub addresses: Vec<&'a Address>,
    /// The topics
    pub topics: Vec<Option<Vec<&'a B256>>>,
}

/// Error type for [`CacheableLogFilterOptions::try_from`] and
/// [`CacheableLogFilterRange::try_from`].
#[derive(thiserror::Error, Debug)]
#[error("Method is not cacheable: {0:?}")]
pub struct LogFilterOptionsNotCacheableError(LogFilterOptions);

impl<'a> TryFrom<&'a LogFilterOptions> for CacheableLogFilterOptions<'a> {
    type Error = LogFilterOptionsNotCacheableError;

    fn try_from(value: &'a LogFilterOptions) -> Result<Self, Self::Error> {
        let range = CacheableLogFilterRange::try_from(value)?;

        Ok(Self {
            range,
            addresses: value
                .address
                .as_ref()
                .map_or(Vec::new(), |address| match address {
                    OneOrMore::One(address) => vec![address],
                    OneOrMore::Many(addresses) => addresses.iter().collect(),
                }),
            topics: value.topics.as_ref().map_or(Vec::new(), |topics| {
                topics
                    .iter()
                    .map(|options| {
                        options.as_ref().map(|options| match options {
                            OneOrMore::One(topic) => vec![topic],
                            OneOrMore::Many(topics) => topics.iter().collect(),
                        })
                    })
                    .collect()
            }),
        })
    }
}

/// A cacheable range input for the `eth_getLogs` method.
#[derive(Clone, Debug)]
pub enum CacheableLogFilterRange<'a> {
    /// The `block_hash` argument
    Hash(&'a B256),
    /// A range of blocks
    Range {
        /// The `from_block` argument
        from_block: CacheableBlockSpec<'a>,
        /// The `to_block` argument
        to_block: CacheableBlockSpec<'a>,
    },
}

impl CacheKeyVariant for CacheableLogFilterRange<'_> {
    fn cache_key_variant(&self) -> u8 {
        match self {
            CacheableLogFilterRange::Hash(_) => 0,
            CacheableLogFilterRange::Range { .. } => 1,
        }
    }
}

impl<'a> TryFrom<&'a LogFilterOptions> for CacheableLogFilterRange<'a> {
    type Error = LogFilterOptionsNotCacheableError;

    fn try_from(value: &'a LogFilterOptions) -> Result<Self, Self::Error> {
        let map_err = |_| LogFilterOptionsNotCacheableError(value.clone());

        if let Some(from_block) = &value.from_block {
            if let Some(to_block) = &value.to_block {
                if value.block_hash.is_none() {
                    let range = Self::Range {
                        from_block: from_block.try_into().map_err(map_err)?,
                        to_block: to_block.try_into().map_err(map_err)?,
                    };

                    return Ok(range);
                }
            }
        } else if let Some(block_hash) = &value.block_hash {
            if value.from_block.is_none() {
                return Ok(Self::Hash(block_hash));
            }
        }

        Err(LogFilterOptionsNotCacheableError(value.clone()))
    }
}
