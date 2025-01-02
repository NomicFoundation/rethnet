//! Support types for configuring storage caching

use std::{fmt, str::FromStr};

use alloy_chains::Chain;

/// Settings to configure caching of remote
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageCachingConfig {
    /// chains to cache
    pub chains: CachedChains,
    /// endpoints to cache
    pub endpoints: CachedEndpoints,
}

impl StorageCachingConfig {
    /// Whether caching should be enabled for the endpoint
    pub fn enable_for_endpoint(&self, endpoint: impl AsRef<str>) -> bool {
        self.endpoints.is_match(endpoint)
    }

    /// Whether caching should be enabled for the chain id
    pub fn enable_for_chain_id(&self, chain_id: u64) -> bool {
        // ignore dev chains
        if [99, 1337, 31337].contains(&chain_id) {
            return false;
        }
        self.chains.is_match(chain_id)
    }
}

/// What chains to cache
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CachedChains {
    /// Cache all chains
    #[default]
    All,
    /// Don't cache anything
    None,
    /// Only cache these chains
    Chains(Vec<Chain>),
}
impl CachedChains {
    /// Whether the `endpoint` matches
    pub fn is_match(&self, chain: u64) -> bool {
        match self {
            CachedChains::All => true,
            CachedChains::None => false,
            CachedChains::Chains(chains) => chains.iter().any(|c| c.id() == chain),
        }
    }
}

/// What endpoints to enable caching for
#[derive(Clone, Debug, Default)]
pub enum CachedEndpoints {
    /// Cache all endpoints
    #[default]
    All,
    /// Only cache non-local host endpoints
    Remote,
    /// Only cache these chains
    Pattern(regex::Regex),
}

impl CachedEndpoints {
    /// Whether the `endpoint` matches
    pub fn is_match(&self, endpoint: impl AsRef<str>) -> bool {
        let endpoint = endpoint.as_ref();
        match self {
            CachedEndpoints::All => true,
            CachedEndpoints::Remote => {
                !endpoint.contains("localhost:") && !endpoint.contains("127.0.0.1:")
            }
            CachedEndpoints::Pattern(re) => re.is_match(endpoint),
        }
    }
}

impl PartialEq for CachedEndpoints {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CachedEndpoints::Pattern(a), CachedEndpoints::Pattern(b)) => a.as_str() == b.as_str(),
            (&CachedEndpoints::All, &CachedEndpoints::All)
            | (&CachedEndpoints::Remote, &CachedEndpoints::Remote) => true,
            _ => false,
        }
    }
}

impl Eq for CachedEndpoints {}

impl fmt::Display for CachedEndpoints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CachedEndpoints::All => f.write_str("all"),
            CachedEndpoints::Remote => f.write_str("remote"),
            CachedEndpoints::Pattern(s) => s.fmt(f),
        }
    }
}

impl FromStr for CachedEndpoints {
    type Err = regex::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "all" => Ok(CachedEndpoints::All),
            "remote" => Ok(CachedEndpoints::Remote),
            _ => Ok(CachedEndpoints::Pattern(s.parse()?)),
        }
    }
}
