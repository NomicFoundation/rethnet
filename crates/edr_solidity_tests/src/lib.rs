#[macro_use]
extern crate tracing;

use std::{
    collections::HashSet,
    fmt::Debug,
    sync::{OnceLock, RwLock},
};

use proptest::test_runner::{
    FailurePersistence, FileFailurePersistence, RngAlgorithm, TestRng, TestRunner,
};

pub mod coverage;

pub mod gas_report;

pub mod multi_runner;
pub use multi_runner::MultiContractRunner;

mod runner;
pub use runner::ContractRunner;

mod config;
pub use config::{SolidityTestRunnerConfig, SolidityTestRunnerConfigError};

pub mod result;
mod test_filter;
pub use foundry_evm::*;
use foundry_evm::{executors::fuzz::FuzzConfig, fuzz::invariant::InvariantConfig};
pub use test_filter::TestFilter;

static FAILURE_PATHS: OnceLock<RwLock<HashSet<&'static str>>> = OnceLock::new();

/// Metadata on how to run fuzz/invariant tests
#[derive(Clone, Debug, Default)]
pub struct TestOptions {
    /// The base "fuzz" test configuration
    pub fuzz: FuzzConfig,
    /// The base "invariant" test configuration
    pub invariant: InvariantConfig,
}

impl TestOptions {
    /// Returns a "fuzz" test runner instance.
    pub fn fuzz_runner(&self) -> TestRunner {
        let fuzz_config = self.fuzz_config().clone();

        if let Some(failure_persist_dir) = fuzz_config.failure_persist_dir {
            let failure_persist_path = failure_persist_dir
                .join(fuzz_config.failure_persist_file)
                .into_os_string()
                .into_string()
                .expect("path should be valid UTF-8");

            // HACK: We need to leak the path as
            // `proptest::test_runner::FileFailurePersistence` requires a
            // `&'static str`. We mitigate this by making sure that one particular path
            // is only leaked once.
            let failure_paths = FAILURE_PATHS.get_or_init(RwLock::default);
            // Need to be in a block to ensure that the read lock is dropped before we try
            // to insert.
            {
                let failure_paths_guard = failure_paths.read().expect("lock is not poisoned");
                if let Some(static_path) = failure_paths_guard.get(&*failure_persist_path) {
                    return self.fuzzer_with_cases(
                        fuzz_config.runs,
                        Some(Box::new(FileFailurePersistence::Direct(static_path))),
                    );
                }
            }
            // Write block
            {
                let mut failure_paths_guard = failure_paths.write().expect("lock is not poisoned");
                failure_paths_guard.insert(failure_persist_path.clone().leak());
                let static_path = failure_paths_guard
                    .get(&*failure_persist_path)
                    .expect("must exist since we just inserted it");

                self.fuzzer_with_cases(
                    fuzz_config.runs,
                    Some(Box::new(FileFailurePersistence::Direct(static_path))),
                )
            }
        } else {
            self.fuzzer_with_cases(fuzz_config.runs, None)
        }
    }

    /// Returns an "invariant" test runner instance.
    pub fn invariant_runner(&self) -> TestRunner {
        let invariant = self.invariant_config();
        self.fuzzer_with_cases(invariant.runs, None)
    }

    /// Returns a "fuzz" configuration setup.
    pub fn fuzz_config(&self) -> &FuzzConfig {
        &self.fuzz
    }

    /// Returns an "invariant" configuration setup.
    pub fn invariant_config(&self) -> &InvariantConfig {
        &self.invariant
    }

    pub fn fuzzer_with_cases(
        &self,
        cases: u32,
        file_failure_persistence: Option<Box<dyn FailurePersistence>>,
    ) -> TestRunner {
        let config = proptest::test_runner::Config {
            failure_persistence: file_failure_persistence,
            cases,
            max_global_rejects: self.fuzz.max_test_rejects,
            ..Default::default()
        };

        if let Some(seed) = &self.fuzz.seed {
            trace!(target: "edr_solidity_tests::test", %seed, "building deterministic fuzzer");
            let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &seed.to_be_bytes::<32>());
            TestRunner::new_with_rng(config, rng)
        } else {
            trace!(target: "edr_solidity_tests::test", "building stochastic fuzzer");
            TestRunner::new(config)
        }
    }
}
