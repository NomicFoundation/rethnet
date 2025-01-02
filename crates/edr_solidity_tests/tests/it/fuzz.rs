//! Fuzz tests.

use std::collections::BTreeMap;

use alloy_primitives::{Bytes, U256};
use edr_solidity_tests::{
    fuzz::CounterExample,
    result::{SuiteResult, TestStatus},
};

use crate::helpers::{assert_multiple, SolidityTestFilter, TestFuzzConfig, TEST_DATA_DEFAULT};

#[tokio::test(flavor = "multi_thread")]
async fn test_fuzz() {
    let filter = SolidityTestFilter::new(".*", ".*", ".*fuzz/")
        .exclude_tests(r"invariantCounter|testIncrement\(address\)|testNeedle\(uint256\)|testSuccessChecker\(uint256\)|testSuccessChecker2\(int256\)|testSuccessChecker3\(uint32\)")
        .exclude_paths("invariant");
    let runner = TEST_DATA_DEFAULT.runner().await;
    let suite_result = runner.test_collect(filter).await;

    assert!(!suite_result.is_empty());

    for (_, SuiteResult { test_results, .. }) in suite_result {
        for (test_name, result) in test_results {
            match test_name.as_str() {
                "testPositive(uint256)"
                | "testPositive(int256)"
                | "testSuccessfulFuzz(uint128,uint128)"
                | "testToStringFuzz(bytes32)" => assert_eq!(
                    result.status,
                    TestStatus::Success,
                    "Test {} did not pass as expected.\nReason: {:?}\nLogs:\n{}",
                    test_name,
                    result.reason,
                    result.decoded_logs.join("\n")
                ),
                _ => assert_eq!(
                    result.status,
                    TestStatus::Failure,
                    "Test {} did not fail as expected.\nReason: {:?}\nLogs:\n{}",
                    test_name,
                    result.reason,
                    result.decoded_logs.join("\n")
                ),
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_successful_fuzz_cases() {
    let filter = SolidityTestFilter::new(".*", ".*", ".*fuzz/FuzzPositive")
        .exclude_tests(r"invariantCounter|testIncrement\(address\)|testNeedle\(uint256\)")
        .exclude_paths("invariant");
    let runner = TEST_DATA_DEFAULT.runner().await;
    let suite_result = runner.test_collect(filter).await;

    assert!(!suite_result.is_empty());

    for (_, SuiteResult { test_results, .. }) in suite_result {
        for (test_name, result) in test_results {
            match test_name.as_str() {
                "testSuccessChecker(uint256)"
                | "testSuccessChecker2(int256)"
                | "testSuccessChecker3(uint32)" => assert_eq!(
                    result.status,
                    TestStatus::Success,
                    "Test {} did not pass as expected.\nReason: {:?}\nLogs:\n{}",
                    test_name,
                    result.reason,
                    result.decoded_logs.join("\n")
                ),
                _ => {}
            }
        }
    }
}

/// Test that showcases PUSH collection on normal fuzzing. Ignored until we
/// collect them in a smarter way.
/// Disabled in https://github.com/foundry-rs/foundry/pull/2724
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_fuzz_collection() {
    let filter = SolidityTestFilter::new(".*", ".*", ".*fuzz/FuzzCollection.t.sol");
    let mut config = TEST_DATA_DEFAULT.base_runner_config();
    config.invariant.depth = 100;
    config.invariant.runs = 1000;
    config.fuzz.runs = 1000;
    config.fuzz.seed = Some(U256::from(6u32));
    let runner = TEST_DATA_DEFAULT.runner_with_config(config).await;
    let results = runner.test_collect(filter).await;

    assert_multiple(
        &results,
        BTreeMap::from([(
            "default/fuzz/FuzzCollection.t.sol:SampleContractTest",
            vec![
                (
                    "invariantCounter",
                    false,
                    Some("broken counter.".into()),
                    None,
                    None,
                ),
                (
                    "testIncrement(address)",
                    false,
                    Some("Call did not revert as expected".into()),
                    None,
                    None,
                ),
                (
                    "testNeedle(uint256)",
                    false,
                    Some("needle found.".into()),
                    None,
                    None,
                ),
            ],
        )]),
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_persist_fuzz_failure() {
    let filter = SolidityTestFilter::new(".*", ".*", ".*fuzz/FuzzFailurePersist.t.sol");
    let mut fuzz_config = TestFuzzConfig {
        runs: 1000,
        ..TestFuzzConfig::default()
    };
    let runner = TEST_DATA_DEFAULT
        .runner_with_fuzz_config(fuzz_config.clone())
        .await;

    macro_rules! get_failure_result {
        ($runner:ident) => {
            $runner
                .clone()
                .test_collect(filter.clone()).await
                .get("default/fuzz/FuzzFailurePersist.t.sol:FuzzFailurePersistTest")
                .unwrap()
                .test_results
                .get("test_persist_fuzzed_failure(uint256,int256,address,bool,string,(address,uint256),address[])")
                .unwrap()
                .counterexample
                .clone()
        };
    }

    // record initial counterexample calldata
    let initial_counterexample = get_failure_result!(runner);
    let initial_calldata = match initial_counterexample {
        Some(CounterExample::Single(counterexample)) => counterexample.calldata,
        _ => Bytes::new(),
    };

    // run several times and compare counterexamples calldata
    for i in 0..10 {
        let new_calldata = match get_failure_result!(runner) {
            Some(CounterExample::Single(counterexample)) => counterexample.calldata,
            _ => Bytes::new(),
        };
        // calldata should be the same with the initial one
        assert_eq!(initial_calldata, new_calldata, "run {i}");
    }

    // write new failure in different file, but keep the same directory
    fuzz_config.failure_persist_file = "failure1".to_string();
    let runner = TEST_DATA_DEFAULT.runner_with_fuzz_config(fuzz_config).await;
    let new_calldata = match get_failure_result!(runner) {
        Some(CounterExample::Single(counterexample)) => counterexample.calldata,
        _ => Bytes::new(),
    };
    // empty file is used to load failure so new calldata is generated
    assert_ne!(initial_calldata, new_calldata);
}
