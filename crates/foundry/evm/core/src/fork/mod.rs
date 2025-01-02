use std::path::PathBuf;

use revm::primitives::Env;

use super::opts::EvmOpts;

mod init;
use alloy_chains::Chain;
pub use init::environment;

pub mod database;

mod multi;
pub mod provider;

pub use multi::{ForkId, MultiFork, MultiForkHandler};

/// Represents a _fork_ of a remote chain whose data is available only via the
/// `url` endpoint.
#[derive(Clone, Debug)]
pub struct CreateFork {
    /// Optional RPC cache path. If this is none, then no rpc calls will be
    /// cached, otherwise data is cached to `<rpc_cache_path>/<chain id>/<block
    /// number>`.
    pub rpc_cache_path: Option<PathBuf>,
    /// The URL to a node for fetching remote state
    pub url: String,
    /// The env to create this fork, main purpose is to provide some metadata
    /// for the fork
    pub env: Env,
    /// All env settings as configured by the user
    pub evm_opts: EvmOpts,
}

impl CreateFork {
    /// Returns the path to the cache dir of the `block` on the `chain`
    /// based on the configured rpc cache path.
    pub fn block_cache_dir(&self, chain_id: impl Into<Chain>, block: u64) -> Option<PathBuf> {
        self.rpc_cache_path.as_ref().map(|rpc_cache_path| {
            rpc_cache_path
                .join(chain_id.into().to_string())
                .join(block.to_string())
        })
    }
}
