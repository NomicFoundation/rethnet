use std::path::PathBuf;

use alloy_primitives::{Address, B256};
use foundry_evm::{
    executors::{fuzz::FuzzConfig, invariant::InvariantConfig},
    inspectors::cheatcodes::CheatsConfigOptions,
};

use crate::{
    fork::CreateFork,
    opts::{Env as EvmEnv, EvmOpts},
    revm::primitives::{SpecId, U256},
};

#[derive(Debug, thiserror::Error)]
pub enum SolidityTestRunnerConfigError {
    /// Failed to create a fork with the given config.
    #[error("{0}")]
    CreateFork(eyre::Error),
    /// Failed to create an EVM environment with the given config.
    #[error("{0}")]
    EvmEnv(eyre::Error),
    /// Failed to normalize project root
    #[error("Failed to normalize project root with error: {0}")]
    InvalidProjectRoot(std::io::Error),
}

/// Solidity tests configuration
#[derive(Clone, Debug)]
pub struct SolidityTestRunnerConfig {
    /// Project root directory.
    pub project_root: PathBuf,
    /// Whether to enable trace mode.
    pub trace: bool,
    /// Whether to collect coverage info
    pub coverage: bool,
    /// Whether to support the `testFail` prefix
    pub test_fail: bool,
    /// Whether to enable solidity fuzz fixtures support
    pub solidity_fuzz_fixtures: bool,
    /// Cheats configuration options
    pub cheats_config_options: CheatsConfigOptions,
    /// EVM options
    pub evm_opts: EvmOpts,
    /// Configuration for fuzz testing
    pub fuzz: FuzzConfig,
    /// Configuration for invariant testing
    pub invariant: InvariantConfig,
}

impl SolidityTestRunnerConfig {
    /// The default evm options for the Solidity test runner.
    pub fn default_evm_opts() -> EvmOpts {
        EvmOpts {
            env: EvmEnv {
                gas_limit: i64::MAX.try_into().expect("max i64 fits into u64"),
                chain_id: Some(31337),
                gas_price: Some(0),
                block_base_fee_per_gas: 0,
                tx_origin: edr_defaults::SOLIDITY_TESTS_SENDER,
                block_number: 1,
                block_difficulty: 0,
                block_prevrandao: B256::default(),
                block_gas_limit: None,
                block_timestamp: 1,
                block_coinbase: Address::default(),
                code_size_limit: None,
            },
            spec: SpecId::CANCUN,
            fork_url: None,
            fork_block_number: None,
            fork_retries: None,
            fork_retry_backoff: None,
            compute_units_per_second: None,
            no_rpc_rate_limit: false,
            sender: edr_defaults::SOLIDITY_TESTS_SENDER,
            initial_balance: U256::from(0xffffffffffffffffffffffffu128),
            ffi: false,
            always_use_create_2_factory: false,
            memory_limit: 1 << 25, // 2**25 = 32MiB
            isolate: false,
            disable_block_gas_limit: false,
        }
    }

    pub async fn get_fork(&self) -> Result<Option<CreateFork>, SolidityTestRunnerConfigError> {
        if let Some(fork_url) = self.evm_opts.fork_url.as_ref() {
            let evm_env = self
                .evm_opts
                .fork_evm_env(fork_url)
                .await
                .map_err(SolidityTestRunnerConfigError::CreateFork)?
                .0;

            let rpc_cache_path = self.rpc_cache_path(fork_url, evm_env.cfg.chain_id);

            Ok(Some(CreateFork {
                rpc_cache_path,
                url: fork_url.clone(),
                env: evm_env,
                evm_opts: self.evm_opts.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Whether caching should be enabled for the given chain id
    fn rpc_cache_path(&self, endpoint: &str, chain_id: impl Into<u64>) -> Option<PathBuf> {
        let enable_for_chain_id = self
            .cheats_config_options
            .rpc_storage_caching
            .enable_for_chain_id(chain_id.into());
        let enable_for_endpoint = self
            .cheats_config_options
            .rpc_storage_caching
            .enable_for_endpoint(endpoint);
        if enable_for_chain_id && enable_for_endpoint {
            self.cheats_config_options.rpc_cache_path.clone()
        } else {
            None
        }
    }
}
