//! Test helpers for Forge integration tests.

mod config;
pub use config::{assert_multiple, TestConfig};
mod integration_test_config;
mod linker;
mod solidity_error_code;
mod solidity_test_filter;
pub use solidity_test_filter::SolidityTestFilter;
mod tracing;

use std::{borrow::Cow, env, fmt, io::Write, path::PathBuf};

use alloy_primitives::U256;
use edr_solidity_tests::{
    fuzz::FuzzDictionaryConfig,
    multi_runner::{TestContract, TestContracts},
    revm::primitives::SpecId,
    MultiContractRunner, SolidityTestRunnerConfig,
};
use edr_test_utils::{
    env::{get_alchemy_url_for_network, NetworkType},
    new_fd_lock,
};
use foundry_cheatcodes::{ExecutionContextConfig, FsPermissions, RpcEndpoint, RpcEndpoints};
use foundry_compilers::{
    artifacts::{CompactContractBytecode, Libraries},
    Artifact, EvmVersion, Project, ProjectCompileOutput,
};
use foundry_evm::{
    abi::TestFunctionExt,
    constants::CALLER,
    contracts::ContractsByArtifact,
    decode::RevertDecoder,
    executors::invariant::InvariantConfig,
    fuzz::FuzzConfig,
    inspectors::cheatcodes::CheatsConfigOptions,
    opts::{Env, EvmOpts},
};
use linker::{LinkOutput, Linker};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::helpers::{
    integration_test_config::IntegrationTestConfig, tracing::init_tracing_for_solidity_tests,
};

pub const RE_PATH_SEPARATOR: &str = "/";
static PROJECT_ROOT: Lazy<PathBuf> = Lazy::new(|| {
    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/testdata");
    dunce::canonicalize(PathBuf::from(TESTDATA)).expect("Failed to canonicalize root")
});

/// Profile for the tests group. Used to configure separate configurations for
/// test runs.
pub enum ForgeTestProfile {
    Default,
    Cancun,
    MultiVersion,
}

impl fmt::Display for ForgeTestProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForgeTestProfile::Default => write!(f, "default"),
            ForgeTestProfile::Cancun => write!(f, "cancun"),
            ForgeTestProfile::MultiVersion => write!(f, "multi-version"),
        }
    }
}

impl ForgeTestProfile {
    /// Returns true if the profile is Cancun.
    fn is_cancun(&self) -> bool {
        matches!(self, Self::Cancun)
    }

    fn project(&self) -> Project {
        self.integration_test_config()
            .project()
            .expect("Failed to build project")
    }

    fn evm_opts() -> EvmOpts {
        EvmOpts {
            env: Env {
                gas_limit: u64::MAX,
                chain_id: None,
                tx_origin: CALLER,
                block_number: 1,
                block_timestamp: 1,
                ..Env::default()
            },
            sender: CALLER,
            initial_balance: U256::MAX,
            ffi: true,
            memory_limit: 1 << 26,
            spec: SpecId::CANCUN,
            ..EvmOpts::default()
        }
    }

    fn runner_config() -> SolidityTestRunnerConfig {
        SolidityTestRunnerConfig {
            trace: true,
            evm_opts: Self::evm_opts(),
            project_root: PROJECT_ROOT.clone(),
            cheats_config_options: CheatsConfigOptions {
                execution_context: ExecutionContextConfig::Test,
                ..CheatsConfigOptions::default()
            },
            fuzz: TestFuzzConfig::default().into(),
            invariant: TestInvariantConfig::default().into(),
            coverage: false,
            test_fail: true,
            solidity_fuzz_fixtures: true,
        }
    }

    /// Build [`IntegrationTestConfig`] for test profile.
    ///
    /// Project source files are read from `testdata/{profile_name`}
    /// Project output files are written to `testdata/out/{profile_name`}
    /// Cache is written to `testdata/cache/{profile_name`}
    ///
    /// AST output is enabled by default to support inline configs.
    fn integration_test_config(&self) -> IntegrationTestConfig {
        let mut config = IntegrationTestConfig::with_root(PROJECT_ROOT.clone());

        config.ast = true;
        config.src = PROJECT_ROOT.join(self.to_string());
        config.out = PROJECT_ROOT.join("out").join(self.to_string());
        config.cache_path = PROJECT_ROOT.join("cache").join(self.to_string());
        config.libraries = vec![
            "fork/Fork.t.sol:DssExecLib:0xfD88CeE74f7D78697775aBDAE53f9Da1559728E4".to_string(),
        ];

        if self.is_cancun() {
            config.evm_version = EvmVersion::Cancun;
        }

        config
    }
}

/// Fuzz testing config with different defaults than
/// [`foundry_config::FuzzConfig`]. See [`foundry_config::FuzzConfig`] for
/// documentation.
#[derive(Debug, Clone)]
pub struct TestFuzzConfig {
    pub runs: u32,
    pub max_test_rejects: u32,
    pub seed: Option<U256>,
    pub dictionary: TestFuzzDictionaryConfig,
    pub gas_report_samples: u32,
    pub failure_persist_dir: Option<PathBuf>,
    pub failure_persist_file: String,
}

impl Default for TestFuzzConfig {
    fn default() -> Self {
        TestFuzzConfig {
            runs: 256,
            max_test_rejects: 65536,
            seed: None,
            dictionary: TestFuzzDictionaryConfig::default(),
            gas_report_samples: 256,
            failure_persist_dir: Some(tempfile::tempdir().unwrap().into_path()),
            failure_persist_file: "testfailure".into(),
        }
    }
}

impl From<TestFuzzConfig> for FuzzConfig {
    fn from(value: TestFuzzConfig) -> Self {
        FuzzConfig {
            runs: value.runs,
            max_test_rejects: value.max_test_rejects,
            seed: value.seed,
            dictionary: value.dictionary.into(),
            gas_report_samples: value.gas_report_samples,
            failure_persist_dir: value.failure_persist_dir,
            failure_persist_file: value.failure_persist_file,
        }
    }
}

/// Fuzz testing config with different defaults than
/// [`foundry_config::InvariantConfig`]. See [`foundry_config::InvariantConfig`]
/// for documentation.
#[derive(Debug, Clone)]
pub struct TestInvariantConfig {
    pub runs: u32,
    pub depth: u32,
    pub fail_on_revert: bool,
    pub call_override: bool,
    pub dictionary: FuzzDictionaryConfig,
    pub shrink_run_limit: usize,
    pub max_assume_rejects: u32,
    pub gas_report_samples: u32,
    pub failure_persist_dir: Option<PathBuf>,
}

impl Default for TestInvariantConfig {
    fn default() -> Self {
        TestInvariantConfig {
            runs: 256,
            depth: 15,
            fail_on_revert: false,
            call_override: false,
            dictionary: FuzzDictionaryConfig {
                dictionary_weight: 80,
                include_storage: true,
                include_push_bytes: true,
                max_fuzz_dictionary_addresses: 10_000,
                max_fuzz_dictionary_values: 10_000,
            },
            shrink_run_limit: 2usize.pow(18u32),
            max_assume_rejects: 65536,
            gas_report_samples: 256,
            failure_persist_dir: Some(tempfile::tempdir().unwrap().into_path()),
        }
    }
}

impl From<TestInvariantConfig> for InvariantConfig {
    fn from(value: TestInvariantConfig) -> Self {
        InvariantConfig {
            runs: value.runs,
            depth: value.depth,
            fail_on_revert: value.fail_on_revert,
            call_override: value.call_override,
            dictionary: value.dictionary,
            shrink_run_limit: value.shrink_run_limit,
            max_assume_rejects: value.max_assume_rejects,
            gas_report_samples: value.gas_report_samples,
            failure_persist_dir: value.failure_persist_dir,
        }
    }
}

/// Fuzz dictionary config with different defaults than
/// [`foundry_config::FuzzDictionaryConfig`].
/// See [`foundry_config::FuzzDictionaryConfig`] for documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestFuzzDictionaryConfig {
    pub dictionary_weight: u32,
    pub include_storage: bool,
    pub include_push_bytes: bool,
    pub max_fuzz_dictionary_addresses: usize,
    pub max_fuzz_dictionary_values: usize,
}

impl Default for TestFuzzDictionaryConfig {
    fn default() -> Self {
        TestFuzzDictionaryConfig {
            dictionary_weight: 40,
            include_storage: true,
            include_push_bytes: true,
            max_fuzz_dictionary_addresses: 10_000,
            max_fuzz_dictionary_values: 10_000,
        }
    }
}

impl From<TestFuzzDictionaryConfig> for FuzzDictionaryConfig {
    fn from(value: TestFuzzDictionaryConfig) -> Self {
        FuzzDictionaryConfig {
            dictionary_weight: value.dictionary_weight,
            include_storage: value.include_storage,
            include_push_bytes: value.include_push_bytes,
            max_fuzz_dictionary_addresses: value.max_fuzz_dictionary_addresses,
            max_fuzz_dictionary_values: value.max_fuzz_dictionary_values,
        }
    }
}

/// Container for test data for a specific test profile.
pub struct ForgeTestData {
    project: Project,
    test_contracts: TestContracts,
    known_contracts: ContractsByArtifact,
    revert_decoder: RevertDecoder,
    runner_config: SolidityTestRunnerConfig,
}

impl ForgeTestData {
    /// Builds [`ForgeTestData`] for the given [`ForgeTestProfile`].
    ///
    /// Uses [`get_compiled`] to lazily compile the project.
    pub fn new(profile: ForgeTestProfile) -> Self {
        let project = profile.project();
        let output = get_compiled(&project);
        let runner_config = ForgeTestProfile::runner_config();

        let root = project.root();
        let output = output.clone().with_stripped_file_prefixes(root);
        let linker = Linker::new(root, output.artifact_ids().collect());

        // Build revert decoder from ABIs of all artifacts.
        let abis = linker
            .contracts
            .iter()
            .filter_map(|(_, contract)| contract.abi.as_ref().map(std::borrow::Borrow::borrow));
        let revert_decoder = RevertDecoder::new().with_abis(abis);

        // Create a mapping of name => (abi, deployment code, Vec<library deployment
        // code>)
        let mut test_contracts = TestContracts::default();

        for (id, contract) in linker.contracts.iter() {
            let Some(abi) = contract.abi.as_ref() else {
                continue;
            };

            // if it's a test, link it and add to deployable contracts
            if abi
                .constructor
                .as_ref()
                .map_or(true, |c| c.inputs.is_empty())
                && abi
                    .functions()
                    .any(|func| func.name.is_test() || func.name.is_invariant_test())
            {
                let LinkOutput {
                    libs_to_deploy,
                    libraries,
                } = linker
                    .link_with_nonce_or_address(
                        Libraries::default(),
                        runner_config.evm_opts.sender,
                        1,
                        id,
                    )
                    .expect("Libraries should be ok");

                let linked_contract = linker.link(id, &libraries).expect("Linking should be ok");

                let Some(bytecode) = linked_contract
                    .get_bytecode_bytes()
                    .map(std::borrow::Cow::into_owned)
                    .filter(|b| !b.is_empty())
                else {
                    continue;
                };

                test_contracts.insert(
                    id.clone().into(),
                    TestContract {
                        abi: abi.clone().into_owned(),
                        bytecode,
                        libs_to_deploy,
                        libraries,
                    },
                );
            }
        }

        let known_contracts = ContractsByArtifact::new_from_foundry_linker(
            linker.contracts.into_iter().map(|(id, contract)| {
                (
                    id,
                    CompactContractBytecode {
                        abi: contract.abi.map(Cow::into_owned),
                        bytecode: contract.bytecode.map(Cow::into_owned),
                        deployed_bytecode: contract.deployed_bytecode.map(Cow::into_owned),
                    },
                )
            }),
        );

        Self {
            project,
            test_contracts,
            known_contracts,
            revert_decoder,
            runner_config,
        }
    }

    /// Builds a base runner config
    pub fn base_runner_config(&self) -> SolidityTestRunnerConfig {
        init_tracing_for_solidity_tests();
        self.runner_config.clone()
    }

    /// Builds a non-tracing runner
    pub async fn runner(&self) -> MultiContractRunner {
        let config = self.base_runner_config();
        self.runner_with_config(config).await
    }

    /// Builds a non-tracing runner with the given config
    pub async fn runner_with_config(
        &self,
        mut config: SolidityTestRunnerConfig,
    ) -> MultiContractRunner {
        config.cheats_config_options.rpc_endpoints = rpc_endpoints();
        // `**/edr-cache` is cached in CI
        config.cheats_config_options.rpc_cache_path =
            Some(self.project.root().join("edr-cache/solidity-tests/rpc"));

        // no prompt testing
        config.cheats_config_options.prompt_timeout = 0;

        self.build_runner(config).await
    }

    /// Builds a non-tracing runner with the given filesystem permissions
    pub async fn runner_with_fs_permissions(
        &self,
        fs_permissions: FsPermissions,
    ) -> MultiContractRunner {
        let mut config = self.base_runner_config();
        config.cheats_config_options.fs_permissions = fs_permissions;
        self.runner_with_config(config).await
    }

    /// Builds a non-tracing runner with the given invariant config
    pub async fn runner_with_fuzz_config(
        &self,
        fuzz_config: TestFuzzConfig,
    ) -> MultiContractRunner {
        let mut config = self.base_runner_config();
        config.fuzz = fuzz_config.into();
        self.runner_with_config(config).await
    }

    /// Builds a non-tracing runner with the given invariant config
    pub async fn runner_with_invariant_config(
        &self,
        invariant_config: TestInvariantConfig,
    ) -> MultiContractRunner {
        let mut config = self.base_runner_config();
        config.invariant = invariant_config.into();
        self.runner_with_config(config).await
    }

    /// Builds a non-tracing runner with the given invariant config and fuzz
    /// seed.
    pub async fn runner_with_invariant_config_and_seed(
        &self,
        seed: U256,
        invariant_config: TestInvariantConfig,
    ) -> MultiContractRunner {
        let mut config = self.base_runner_config();
        config.fuzz.seed = Some(seed);
        config.invariant = invariant_config.into();
        self.runner_with_config(config).await
    }

    /// Builds a tracing runner
    pub async fn tracing_runner(&self) -> MultiContractRunner {
        let mut config = self.base_runner_config();
        config.trace = true;
        self.build_runner(config).await
    }

    /// Builds a runner that runs against forked state
    pub async fn forked_runner(&self, rpc: &str) -> MultiContractRunner {
        let mut config = self.base_runner_config();

        config.evm_opts.fork_url = Some(rpc.to_string());

        self.build_runner(config).await
    }

    async fn build_runner(&self, config: SolidityTestRunnerConfig) -> MultiContractRunner {
        MultiContractRunner::new(
            config,
            self.test_contracts.clone(),
            self.known_contracts.clone(),
            self.revert_decoder.clone(),
        )
        .await
        .expect("Config should be ok")
    }
}

fn get_compiled(project: &Project) -> ProjectCompileOutput {
    let lock_file_path = project.sources_path().join(".lock");
    // Compile only once per test run.
    // We need to use a file lock because `cargo-nextest` runs tests in different
    // processes. This is similar to [`edr_test_utils::util::initialize`],
    // see its comments for more details.
    let mut lock = new_fd_lock(&lock_file_path);
    let read = lock.read().unwrap();
    let out;
    if project.cache_path().exists() && std::fs::read(&lock_file_path).unwrap() == b"1" {
        out = project.compile();
        drop(read);
    } else {
        drop(read);
        let mut write = lock.write().unwrap();
        write.write_all(b"1").unwrap();
        out = project.compile();
        drop(write);
    }

    let out = out.unwrap();
    assert!(!out.has_compiler_errors(), "Compiled with errors:\n{out}");
    out
}

/// Default data for the tests group.
pub static TEST_DATA_DEFAULT: Lazy<ForgeTestData> =
    Lazy::new(|| ForgeTestData::new(ForgeTestProfile::Default));

/// Data for tests requiring Cancun support on Solc and EVM level.
pub static TEST_DATA_CANCUN: Lazy<ForgeTestData> =
    Lazy::new(|| ForgeTestData::new(ForgeTestProfile::Cancun));

/// Data for tests requiring Cancun support on Solc and EVM level.
pub static TEST_DATA_MULTI_VERSION: Lazy<ForgeTestData> =
    Lazy::new(|| ForgeTestData::new(ForgeTestProfile::MultiVersion));

fn rpc_endpoints() -> RpcEndpoints {
    RpcEndpoints::new([
        (
            "rpcAliasFake",
            RpcEndpoint::Url("https://example.com".to_string()),
        ),
        (
            "rpcAliasMainnet",
            RpcEndpoint::Url(get_alchemy_url_for_network(NetworkType::Ethereum)),
        ),
        (
            "rpcAliasSepolia",
            RpcEndpoint::Url(get_alchemy_url_for_network(NetworkType::Sepolia)),
        ),
        (
            "rpcEnvAlias",
            RpcEndpoint::Env("${RPC_ENV_ALIAS}".to_string()),
        ),
        (
            "rpcAliasOptimism",
            RpcEndpoint::Url(get_alchemy_url_for_network(NetworkType::Optimism)),
        ),
        (
            "rpcAliasPolygon",
            RpcEndpoint::Url(get_alchemy_url_for_network(NetworkType::Polygon)),
        ),
        (
            "rpcAliasArbitrum",
            RpcEndpoint::Url(get_alchemy_url_for_network(NetworkType::Arbitrum)),
        ),
    ])
}
