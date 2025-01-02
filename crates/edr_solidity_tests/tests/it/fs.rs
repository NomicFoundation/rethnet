//! Filesystem tests.

use foundry_cheatcodes::{FsPermissions, PathPermission};

use crate::helpers::{SolidityTestFilter, TestConfig, TEST_DATA_DEFAULT};

#[tokio::test(flavor = "multi_thread")]
async fn test_fs_disabled() {
    let runner = TEST_DATA_DEFAULT
        .runner_with_fs_permissions(FsPermissions::new(vec![PathPermission::none("./")]))
        .await;
    let filter = SolidityTestFilter::new(".*", ".*", ".*fs/Disabled");
    TestConfig::with_filter(runner, filter).run().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fs_default() {
    let runner = TEST_DATA_DEFAULT
        .runner_with_fs_permissions(FsPermissions::new(vec![PathPermission::read("./fixtures")]))
        .await;
    let filter = SolidityTestFilter::new(".*", ".*", ".*fs/Default");
    TestConfig::with_filter(runner, filter).run().await;
}
