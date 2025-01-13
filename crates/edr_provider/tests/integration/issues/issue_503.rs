use std::{str::FromStr as _, sync::Arc};

use edr_eth::{
    l1::{self, L1ChainSpec},
    Address, U256,
};
use edr_provider::{
    hardhat_rpc_types::ForkConfig, test_utils::create_test_config_with_fork, time::CurrentTime,
    MethodInvocation, NoopLogger, Provider, ProviderRequest,
};
use edr_solidity::contract_decoder::ContractDecoder;
use edr_test_utils::env::get_alchemy_url;
use tokio::runtime;

// https://github.com/NomicFoundation/edr/issues/503
#[tokio::test(flavor = "multi_thread")]
async fn issue_503() -> anyhow::Result<()> {
    let logger = Box::new(NoopLogger::<L1ChainSpec>::default());
    let subscriber = Box::new(|_event| {});

    let mut config = create_test_config_with_fork(Some(ForkConfig {
        json_rpc_url: get_alchemy_url(),
        block_number: Some(19_909_475),
        http_headers: None,
    }));
    config.hardfork = l1::SpecId::CANCUN;

    let provider = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        config,
        Arc::<ContractDecoder>::default(),
        CurrentTime,
    )?;

    let address = Address::from_str("0xbe9895146f7af43049ca1c1ae358b0541ea49704")?;
    let index =
        U256::from_str("0x4f039c94bc7b6c8e7867b9fbd2890a637837fea1c829f434a649c572b15b2969")?;

    provider.handle_request(ProviderRequest::Single(MethodInvocation::GetStorageAt(
        address, index, None,
    )))?;

    provider.handle_request(ProviderRequest::Single(MethodInvocation::SetStorageAt(
        address,
        index,
        U256::from(1u64),
    )))?;

    Ok(())
}
