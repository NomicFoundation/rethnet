//! Support for handling/identifying selectors.

#![allow(missing_docs)]

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;

const SELECTOR_LOOKUP_URL: &str = "https://api.openchain.xyz/signature-database/v1/lookup";

/// The standard request timeout for API requests
const REQ_TIMEOUT: Duration = Duration::from_secs(15);

/// How many request can time out before we decide this is a spurious connection
const MAX_TIMEDOUT_REQ: usize = 4usize;

/// A client that can request API data from `<https://api.openchain.xyz>`
#[derive(Clone, Debug)]
pub(crate) struct SignEthClient {
    inner: reqwest::Client,
    /// Whether the connection is spurious, or API is down
    spurious_connection: Arc<AtomicBool>,
    /// How many requests timed out
    timedout_requests: Arc<AtomicUsize>,
    /// Max allowed request that can time out
    max_timedout_requests: usize,
}

impl SignEthClient {
    /// Creates a new client with default settings
    pub(crate) fn new() -> reqwest::Result<Self> {
        let inner = reqwest::Client::builder()
            .default_headers(HeaderMap::from_iter([(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("edr_solidity_tests"),
            )]))
            .timeout(REQ_TIMEOUT)
            .build()?;
        Ok(Self {
            inner,
            spurious_connection: Arc::new(AtomicBool::default()),
            timedout_requests: Arc::new(AtomicUsize::default()),
            max_timedout_requests: MAX_TIMEDOUT_REQ,
        })
    }

    async fn get_text(&self, url: &str) -> reqwest::Result<String> {
        trace!(%url, "GET");
        self.inner
            .get(url)
            .send()
            .await
            .map_err(|err| {
                self.on_reqwest_err(&err);
                err
            })?
            .text()
            .await
            .map_err(|err| {
                self.on_reqwest_err(&err);
                err
            })
    }

    fn on_reqwest_err(&self, err: &reqwest::Error) {
        fn is_connectivity_err(err: &reqwest::Error) -> bool {
            if err.is_timeout() || err.is_connect() {
                return true;
            }
            // Error HTTP codes (5xx) are considered connectivity issues and will prompt
            // retry
            if let Some(status) = err.status() {
                let code = status.as_u16();
                if (500..600).contains(&code) {
                    return true;
                }
            }
            false
        }

        if is_connectivity_err(err) {
            warn!("spurious network detected for <https://api.openchain.xyz>");
            let previous = self.timedout_requests.fetch_add(1, Ordering::SeqCst);
            if previous >= self.max_timedout_requests {
                self.set_spurious();
            }
        }
    }

    /// Returns whether the connection was marked as spurious
    fn is_spurious(&self) -> bool {
        self.spurious_connection.load(Ordering::Relaxed)
    }

    /// Marks the connection as spurious
    fn set_spurious(&self) {
        self.spurious_connection.store(true, Ordering::Relaxed);
    }

    fn ensure_not_spurious(&self) -> eyre::Result<()> {
        if self.is_spurious() {
            eyre::bail!("Spurious connection detected")
        }
        Ok(())
    }

    /// Decodes the given function or event selectors using <https://api.openchain.xyz>
    pub(crate) async fn decode_selectors(
        &self,
        selector_type: SelectorType,
        selectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> eyre::Result<Vec<Option<Vec<String>>>> {
        #[derive(Deserialize)]
        struct Decoded {
            name: String,
        }

        #[derive(Deserialize)]
        struct ApiResult {
            event: HashMap<String, Option<Vec<Decoded>>>,
            function: HashMap<String, Option<Vec<Decoded>>>,
        }

        #[derive(Deserialize)]
        struct ApiResponse {
            ok: bool,
            result: ApiResult,
        }

        let selectors: Vec<String> = selectors
            .into_iter()
            .map(Into::into)
            .map(|s| s.to_lowercase())
            .map(|s| {
                if s.starts_with("0x") {
                    s
                } else {
                    format!("0x{s}")
                }
            })
            .collect();

        if selectors.is_empty() {
            return Ok(vec![]);
        }

        debug!(len = selectors.len(), "decoding selectors");
        trace!(?selectors, "decoding selectors");

        // exit early if spurious connection
        self.ensure_not_spurious()?;

        let expected_len = match selector_type {
            SelectorType::Function => 10, // 0x + hex(4bytes)
            SelectorType::Event => 66,    // 0x + hex(32bytes)
        };
        if let Some(s) = selectors.iter().find(|s| s.len() != expected_len) {
            eyre::bail!(
                "Invalid selector {s}: expected {expected_len} characters (including 0x prefix)."
            )
        }

        // using openchain.xyz signature database over 4byte
        // see https://github.com/foundry-rs/foundry/issues/1672
        let url = format!(
            "{SELECTOR_LOOKUP_URL}?{ltype}={selectors_str}",
            ltype = match selector_type {
                SelectorType::Function => "function",
                SelectorType::Event => "event",
            },
            selectors_str = selectors.join(",")
        );

        let res = self.get_text(&url).await?;
        let api_response = match serde_json::from_str::<ApiResponse>(&res) {
            Ok(inner) => inner,
            Err(err) => {
                eyre::bail!("Could not decode response:\n {res}.\nError: {err}")
            }
        };

        if !api_response.ok {
            eyre::bail!("Failed to decode:\n {res}")
        }

        let decoded = match selector_type {
            SelectorType::Function => api_response.result.function,
            SelectorType::Event => api_response.result.event,
        };

        Ok(selectors
            .into_iter()
            .map(|selector| match decoded.get(&selector) {
                Some(Some(r)) => Some(r.iter().map(|d| d.name.clone()).collect()),
                _ => None,
            })
            .collect())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SelectorType {
    Function,
    Event,
}
