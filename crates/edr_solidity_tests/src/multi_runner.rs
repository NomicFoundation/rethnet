//! Forge test runner for multiple contracts.

use std::{collections::BTreeMap, fmt::Debug, path::PathBuf, sync::Arc, time::Instant};

use alloy_json_abi::JsonAbi;
use alloy_primitives::Bytes;
use eyre::Result;
use foundry_compilers::artifacts::Libraries;
use foundry_evm::{
    backend::Backend,
    contracts::{get_contract_name, ArtifactId, ContractsByArtifact},
    decode::RevertDecoder,
    executors::ExecutorBuilder,
    fork::CreateFork,
    inspectors::{cheatcodes::CheatsConfigOptions, CheatsConfig},
    opts::EvmOpts,
    revm,
};
use futures::StreamExt;

use crate::{
    result::SuiteResult, runner::ContractRunnerOptions, ContractRunner, SolidityTestRunnerConfig,
    SolidityTestRunnerConfigError, TestFilter, TestOptions,
};

#[derive(Debug, Clone)]
pub struct TestContract {
    pub abi: JsonAbi,
    pub bytecode: Bytes,
    pub libs_to_deploy: Vec<Bytes>,
    pub libraries: Libraries,
}

impl TestContract {
    /// Creates a new test contract with the given ABI and bytecode.
    /// Library linking isn't supported for Hardhat test suites
    pub fn new_hardhat(abi: JsonAbi, bytecode: Bytes) -> Self {
        Self {
            abi,
            bytecode,
            libs_to_deploy: vec![],
            libraries: Libraries::default(),
        }
    }
}

pub type TestContracts = BTreeMap<ArtifactId, TestContract>;

/// A multi contract runner receives a set of contracts deployed in an EVM
/// instance and proceeds to run all test functions in these contracts.
#[derive(Clone, Debug)]
pub struct MultiContractRunner {
    /// The project root directory.
    project_root: PathBuf,
    /// Test contracts to deploy
    test_contracts: TestContracts,
    /// Known contracts by artifact id
    known_contracts: Arc<ContractsByArtifact>,
    /// Cheats config.
    cheats_config_options: Arc<CheatsConfigOptions>,
    /// The EVM instance used in the test runner
    evm_opts: EvmOpts,
    /// The configured evm
    env: revm::primitives::Env,
    /// Revert decoder. Contains all known errors and their selectors.
    revert_decoder: RevertDecoder,
    /// The fork to use at launch
    fork: Option<CreateFork>,
    /// Whether to collect coverage info
    coverage: bool,
    /// Whether to collect traces
    trace: bool,
    /// Whether to support the `testFail` prefix
    test_fail: bool,
    /// Whether to enable solidity fuzz fixtures support
    solidity_fuzz_fixtures: bool,
    /// Settings related to fuzz and/or invariant tests
    test_options: TestOptions,
}

impl MultiContractRunner {
    /// Creates a new multi contract runner.
    pub async fn new(
        config: SolidityTestRunnerConfig,
        test_contracts: TestContracts,
        known_contracts: ContractsByArtifact,
        revert_decoder: RevertDecoder,
    ) -> Result<MultiContractRunner, SolidityTestRunnerConfigError> {
        let env = config
            .evm_opts
            .evm_env()
            .await
            .map_err(SolidityTestRunnerConfigError::EvmEnv)?;

        let fork = config.get_fork().await?;

        let SolidityTestRunnerConfig {
            trace,
            coverage,
            test_fail,
            evm_opts,
            project_root,
            cheats_config_options,
            fuzz,
            invariant,
            solidity_fuzz_fixtures,
        } = config;

        // Do canonicalization in blocking context.
        // Canonicalization can touch the file system, hence the blocking thread
        let project_root = tokio::task::spawn_blocking(move || {
            dunce::canonicalize(project_root)
                .map_err(SolidityTestRunnerConfigError::InvalidProjectRoot)
        })
        .await
        .expect("Thread shouldn't panic")?;

        let test_options: TestOptions = TestOptions { fuzz, invariant };

        Ok(Self {
            project_root,
            test_contracts,
            known_contracts: Arc::new(known_contracts),
            cheats_config_options: Arc::new(cheats_config_options),
            evm_opts,
            env,
            revert_decoder,
            fork,
            coverage,
            trace,
            test_fail,
            solidity_fuzz_fixtures,
            test_options,
        })
    }

    /// Returns the known contracts.
    pub fn known_contracts(&self) -> &ContractsByArtifact {
        &self.known_contracts
    }

    /// Executes _all_ tests that match the given `filter`.
    ///
    /// The same as [`test`](Self::test), but returns the results instead of
    /// streaming them.
    ///
    /// Note that this method returns only when all tests have been executed.
    pub async fn test_collect(
        self,
        filter: impl TestFilter + 'static,
    ) -> BTreeMap<String, SuiteResult> {
        let (tx_results, mut rx_results) =
            tokio::sync::mpsc::unbounded_channel::<(ArtifactId, SuiteResult)>();

        self.test(Arc::new(filter), tx_results);

        let mut results = BTreeMap::new();

        while let Some((id, result)) = rx_results.recv().await {
            results.insert(id.identifier(), result);
        }

        results
    }

    /// Executes _all_ tests that match the given `filter`.
    ///
    /// The method will immediately return and send the results to the given
    /// channel as they're ready.
    ///
    /// This will create the runtime based on the configured `evm` ops and
    /// create the `Backend` before executing all contracts and their tests
    /// in _parallel_.
    ///
    /// Each Executor gets its own instance of the `Backend`.
    pub fn test(
        mut self,
        filter: Arc<impl TestFilter + 'static>,
        tx: tokio::sync::mpsc::UnboundedSender<(ArtifactId, SuiteResult)>,
    ) {
        trace!("running all tests");

        // The DB backend that serves all the data.
        let db = Backend::spawn(self.fork.take());

        let find_timer = Instant::now();
        let contracts = self
            .matching_contracts(filter.as_ref())
            .map(|(id, contract)| (id.clone(), contract.clone()))
            .collect::<Vec<_>>();
        let find_time = find_timer.elapsed();
        debug!(
            "Found {} test contracts out of {} in {:?}",
            contracts.len(),
            self.test_contracts.len(),
            find_time,
        );

        let this = Arc::new(self);
        let args = contracts
            .into_iter()
            .zip(std::iter::repeat((this, db, filter, tx)));

        let handle = tokio::runtime::Handle::current();
        handle.spawn(async {
            futures::stream::iter(args)
                .for_each_concurrent(
                    Some(num_cpus::get()),
                    |((id, contract), (this, db, filter, tx))| async move {
                        tokio::task::spawn_blocking(move || {
                            let handle = tokio::runtime::Handle::current();
                            let result = this.run_tests(
                                &id,
                                &contract,
                                db.clone(),
                                filter.as_ref(),
                                &handle,
                            );
                            let _ = tx.send((id, result));
                        })
                        .await
                        .expect("failed to join task");
                    },
                )
                .await;
        });
    }

    /// Returns an iterator over all contracts that match the filter.
    fn matching_contracts<'a>(
        &'a self,
        filter: &'a dyn TestFilter,
    ) -> impl Iterator<Item = (&ArtifactId, &TestContract)> {
        self.test_contracts
            .iter()
            .filter(|&(id, _)| matches_contract(id, filter))
    }

    fn run_tests(
        &self,
        artifact_id: &ArtifactId,
        contract: &TestContract,
        db: Backend,
        filter: &dyn TestFilter,
        handle: &tokio::runtime::Handle,
    ) -> SuiteResult {
        let identifier = artifact_id.identifier();
        let mut span_name = identifier.as_str();

        let cheats_config = CheatsConfig::new(
            self.project_root.clone(),
            (*self.cheats_config_options).clone(),
            self.evm_opts.clone(),
            self.known_contracts.clone(),
            Some(artifact_id.version.clone()),
        );

        let executor = ExecutorBuilder::new()
            .inspectors(|stack| {
                stack
                    .cheatcodes(Arc::new(cheats_config))
                    .trace(self.trace)
                    .coverage(self.coverage)
                    .enable_isolation(self.evm_opts.isolate)
            })
            .spec(self.evm_opts.spec)
            .gas_limit(self.evm_opts.gas_limit())
            .build(self.env.clone(), db);

        if !enabled!(tracing::Level::TRACE) {
            span_name = get_contract_name(&identifier);
        }
        let _guard = info_span!("run_tests", name = span_name).entered();

        debug!("start executing all tests in contract");

        let runner = ContractRunner::new(
            &identifier,
            executor,
            contract,
            &self.revert_decoder,
            ContractRunnerOptions {
                initial_balance: self.evm_opts.initial_balance,
                sender: self.evm_opts.sender,
                test_fail: self.test_fail,
                solidity_fuzz_fixtures: self.solidity_fuzz_fixtures,
            },
        );
        let r = runner.run_tests(
            filter,
            &self.test_options,
            self.known_contracts.clone(),
            handle,
        );

        debug!(duration=?r.duration, "executed all tests in contract");

        r
    }
}

fn matches_contract(id: &ArtifactId, filter: &dyn TestFilter) -> bool {
    filter.matches_path(&id.source) && filter.matches_contract(&id.name)
}
