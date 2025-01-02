//! Support for multiple RPC-endpoints

use std::{collections::BTreeMap, fmt, ops::Deref};

/// Container type for API endpoints, like various RPC endpoints
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RpcEndpoints {
    endpoints: BTreeMap<String, RpcEndpointConfig>,
}

// === impl RpcEndpoints ===

impl RpcEndpoints {
    /// Creates a new list of endpoints
    pub fn new(
        endpoints: impl IntoIterator<Item = (impl Into<String>, impl Into<RpcEndpointType>)>,
    ) -> Self {
        Self {
            endpoints: endpoints
                .into_iter()
                .map(|(name, e)| match e.into() {
                    RpcEndpointType::String(url) => (
                        name.into(),
                        RpcEndpointConfig {
                            endpoint: url,
                            ..Default::default()
                        },
                    ),
                    RpcEndpointType::Config(config) => (name.into(), config),
                })
                .collect(),
        }
    }

    /// Returns `true` if this type doesn't contain any endpoints
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }
}

impl Deref for RpcEndpoints {
    type Target = BTreeMap<String, RpcEndpointConfig>;

    fn deref(&self) -> &Self::Target {
        &self.endpoints
    }
}

/// RPC endpoint wrapper type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcEndpointType {
    /// Raw Endpoint url string
    String(RpcEndpoint),
    /// Config object
    Config(RpcEndpointConfig),
}

impl RpcEndpointType {
    /// Returns the string variant
    pub fn as_endpoint_string(&self) -> Option<&RpcEndpoint> {
        match self {
            RpcEndpointType::String(url) => Some(url),
            RpcEndpointType::Config(_) => None,
        }
    }

    /// Returns the config variant
    pub fn as_endpoint_config(&self) -> Option<&RpcEndpointConfig> {
        match self {
            RpcEndpointType::Config(config) => Some(config),
            RpcEndpointType::String(_) => None,
        }
    }
}

impl fmt::Display for RpcEndpointType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpcEndpointType::String(url) => url.fmt(f),
            RpcEndpointType::Config(config) => config.fmt(f),
        }
    }
}

/// Represents a single endpoint
///
/// This type preserves the value as it's stored in the config. If the value is
/// a reference to an env var, then the `Endpoint::Env` var will hold the
/// reference (`${MAIN_NET}`) and _not_ the value of the env var itself.
/// In other words, this type does not resolve env vars when it's being
/// deserialized
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RpcEndpoint {
    /// A raw Url (ws, http)
    Url(String),
    /// An endpoint that contains at least one `${ENV_VAR}` placeholder
    ///
    /// **Note:** this contains the endpoint as is, like `https://eth-mainnet.alchemyapi.io/v2/${API_KEY}` or `${EPC_ENV_VAR}`
    Env(String),
}

// === impl RpcEndpoint ===

impl RpcEndpoint {
    /// Returns the url variant
    pub fn as_url(&self) -> Option<&str> {
        match self {
            RpcEndpoint::Url(url) => Some(url),
            RpcEndpoint::Env(_) => None,
        }
    }

    /// Returns the env variant
    pub fn as_env(&self) -> Option<&str> {
        match self {
            RpcEndpoint::Env(val) => Some(val),
            RpcEndpoint::Url(_) => None,
        }
    }
}

impl fmt::Display for RpcEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpcEndpoint::Url(url) => url.fmt(f),
            RpcEndpoint::Env(var) => var.fmt(f),
        }
    }
}

impl From<RpcEndpoint> for RpcEndpointType {
    fn from(endpoint: RpcEndpoint) -> Self {
        RpcEndpointType::String(endpoint)
    }
}

impl From<RpcEndpoint> for RpcEndpointConfig {
    fn from(endpoint: RpcEndpoint) -> Self {
        RpcEndpointConfig {
            endpoint,
            ..Default::default()
        }
    }
}

/// Rpc endpoint configuration variant
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcEndpointConfig {
    /// endpoint url or env
    pub endpoint: RpcEndpoint,

    /// The number of retries.
    pub retries: Option<u32>,

    /// Initial retry backoff.
    pub retry_backoff: Option<u64>,

    /// The available compute units per second.
    ///
    /// See also <https://docs.alchemy.com/reference/compute-units#what-are-cups-compute-units-per-second>
    pub compute_units_per_second: Option<u64>,
}

impl fmt::Display for RpcEndpointConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let RpcEndpointConfig {
            endpoint,
            retries,
            retry_backoff,
            compute_units_per_second,
        } = self;

        write!(f, "{endpoint}")?;

        if let Some(retries) = retries {
            write!(f, ", retries={retries}")?;
        }

        if let Some(retry_backoff) = retry_backoff {
            write!(f, ", retry_backoff={retry_backoff}")?;
        }

        if let Some(compute_units_per_second) = compute_units_per_second {
            write!(f, ", compute_units_per_second={compute_units_per_second}")?;
        }

        Ok(())
    }
}

impl From<RpcEndpointConfig> for RpcEndpointType {
    fn from(config: RpcEndpointConfig) -> Self {
        RpcEndpointType::Config(config)
    }
}

impl Default for RpcEndpointConfig {
    fn default() -> Self {
        Self {
            endpoint: RpcEndpoint::Url("http://localhost:8545".to_string()),
            retries: None,
            retry_backoff: None,
            compute_units_per_second: None,
        }
    }
}
