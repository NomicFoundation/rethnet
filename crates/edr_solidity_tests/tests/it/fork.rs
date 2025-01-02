//! Forge forking tests.

use edr_solidity_tests::result::SuiteResult;
use foundry_cheatcodes::{FsPermissions, PathPermission};

use crate::helpers::{SolidityTestFilter, TestConfig, RE_PATH_SEPARATOR, TEST_DATA_DEFAULT};

/// Executes reverting fork test
#[tokio::test(flavor = "multi_thread")]
async fn test_cheats_fork_revert() {
    let filter = SolidityTestFilter::new(
        "testNonExistingContractRevert",
        ".*",
        &format!(".*cheats{RE_PATH_SEPARATOR}Fork2"),
    );
    let runner = TEST_DATA_DEFAULT.runner().await;
    let suite_result = runner.test_collect(filter).await;
    assert_eq!(suite_result.len(), 1);

    for (_, SuiteResult { test_results, .. }) in suite_result {
        for (_, result) in test_results {
            assert_eq!(
                result.reason.unwrap(),
                "Contract 0x5615dEB798BB3E4dFa0139dFa1b3D433Cc23b72f does not exist on active fork with id `1`\n        But exists on non active forks: `[0]`"
            );
        }
    }
}

/// Executes all non-reverting fork cheatcodes
#[tokio::test(flavor = "multi_thread")]
async fn test_cheats_fork() {
    let runner = TEST_DATA_DEFAULT
        .runner_with_fs_permissions(FsPermissions::new(vec![PathPermission::read("./fixtures")]))
        .await;
    let filter = SolidityTestFilter::new(
        "testBlockNumbersMismatch()",
        ".*",
        &format!(".*cheats{RE_PATH_SEPARATOR}Fork"),
    )
    .exclude_tests(".*Revert");
    TestConfig::with_filter(runner, filter).run().await;
}

/// Executes eth_getLogs cheatcode
#[tokio::test(flavor = "multi_thread")]
async fn test_get_logs_fork() {
    let runner = TEST_DATA_DEFAULT
        .runner_with_fs_permissions(FsPermissions::new(vec![PathPermission::read("./fixtures")]))
        .await;
    let filter = SolidityTestFilter::new(
        "testEthGetLogs",
        ".*",
        &format!(".*cheats{RE_PATH_SEPARATOR}Fork"),
    )
    .exclude_tests(".*Revert");
    TestConfig::with_filter(runner, filter).run().await;
}

/// Executes rpc cheatcode
#[tokio::test(flavor = "multi_thread")]
async fn test_rpc_fork() {
    let runner = TEST_DATA_DEFAULT
        .runner_with_fs_permissions(FsPermissions::new(vec![PathPermission::read("./fixtures")]))
        .await;
    let filter =
        SolidityTestFilter::new("testRpc", ".*", &format!(".*cheats{RE_PATH_SEPARATOR}Fork"))
            .exclude_tests(".*Revert");
    TestConfig::with_filter(runner, filter).run().await;
}

/// Tests that we can transact transactions in forking mode
#[tokio::test(flavor = "multi_thread")]
async fn test_transact_fork() {
    let runner = TEST_DATA_DEFAULT.runner().await;
    let filter = SolidityTestFilter::new(".*", ".*", &format!(".*fork{RE_PATH_SEPARATOR}Transact"));
    TestConfig::with_filter(runner, filter).run().await;
}

/// Tests that we can create the same fork (provider,block) concurretnly in
/// different tests
#[tokio::test(flavor = "multi_thread")]
async fn test_create_same_fork() {
    let runner = TEST_DATA_DEFAULT.runner().await;
    let filter = SolidityTestFilter::new(".*", ".*", &format!(".*fork{RE_PATH_SEPARATOR}ForkSame"));
    TestConfig::with_filter(runner, filter).run().await;
}

#[cfg(feature = "test-remote")]
mod remote {
    use super::*;

    /// Tests that we can launch in forking mode
    #[tokio::test(flavor = "multi_thread")]
    async fn test_launch_fork() {
        let rpc_url = edr_test_utils::env::get_alchemy_url();
        let runner = TEST_DATA_DEFAULT.forked_runner(&rpc_url).await;
        let filter =
            SolidityTestFilter::new(".*", ".*", &format!(".*fork{RE_PATH_SEPARATOR}Launch"));
        TestConfig::with_filter(runner, filter).run().await;
    }

    /// Smoke test that forking workings with websockets
    #[tokio::test(flavor = "multi_thread")]
    async fn test_launch_fork_ws() {
        let rpc_url = edr_test_utils::env::get_alchemy_url().replace("https://", "wss://");
        let runner = TEST_DATA_DEFAULT.forked_runner(&rpc_url).await;
        let filter =
            SolidityTestFilter::new(".*", ".*", &format!(".*fork{RE_PATH_SEPARATOR}Launch"));
        TestConfig::with_filter(runner, filter).run().await;
    }
}
