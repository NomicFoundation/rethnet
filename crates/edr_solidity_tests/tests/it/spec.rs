//! Integration tests for EVM specifications.

use foundry_evm::revm::primitives::SpecId;

use crate::helpers::{SolidityTestFilter, TestConfig, TEST_DATA_DEFAULT};

#[tokio::test(flavor = "multi_thread")]
async fn test_shanghai_compat() {
    let filter = SolidityTestFilter::new("", "ShanghaiCompat", ".*spec");
    let mut config = TEST_DATA_DEFAULT.base_runner_config();
    config.evm_opts.spec = SpecId::SHANGHAI;
    TestConfig::with_filter(TEST_DATA_DEFAULT.runner_with_config(config).await, filter)
        .run()
        .await;
}
