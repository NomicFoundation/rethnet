//! Allow forking blocks with future timestamps.
//!
//! See <https://github.com/NomicFoundation/edr/issues/588>

use std::sync::Arc;

use edr_eth::l1::L1ChainSpec;
use edr_provider::{
    hardhat_rpc_types::ForkConfig, test_utils::create_test_config_with_fork, time::MockTime,
    NoopLogger, Provider,
};
use edr_solidity::contract_decoder::ContractDecoder;
use edr_test_utils::env::get_alchemy_url;
use tokio::runtime;

#[tokio::test(flavor = "multi_thread")]
async fn issue_588() -> anyhow::Result<()> {
    let logger = Box::new(NoopLogger::<L1ChainSpec>::default());
    let subscriber = Box::new(|_event| {});

    let early_mainnet_fork = create_test_config_with_fork(Some(ForkConfig {
        json_rpc_url: get_alchemy_url(),
        block_number: Some(2_675_000),
        http_headers: None,
    }));

    let current_time_is_1970 = Arc::new(MockTime::with_seconds(0));

    let _forking_succeeds = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        early_mainnet_fork,
        Arc::<ContractDecoder>::default(),
        current_time_is_1970,
    )?;

    Ok(())
}
