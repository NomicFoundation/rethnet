use std::sync::Arc;

use edr_eth::{address, bytes, Address, BlockSpec, U64};
use edr_optimism::{OptimismChainSpec, OptimismSpecId};
use edr_provider::{
    hardhat_rpc_types::ForkConfig,
    test_utils::{create_test_config_with_fork, ProviderTestFixture},
    time::CurrentTime,
    MethodInvocation, NoopLogger, Provider, ProviderRequest,
};
use edr_rpc_eth::CallRequest;
use edr_solidity::contract_decoder::ContractDecoder;
use edr_test_utils::env::get_alchemy_url;
use tokio::runtime;

const SEPOLIA_CHAIN_ID: u64 = 11_155_420;

fn sepolia_url() -> String {
    get_alchemy_url()
        .replace("eth-", "opt-")
        .replace("mainnet", "sepolia")
}

#[test]
fn sepolia_hardfork_activations() -> anyhow::Result<()> {
    const CANYON_BLOCK_NUMBER: u64 = 4_089_330;

    let url = sepolia_url();
    let fixture = ProviderTestFixture::<OptimismChainSpec>::new_forked(Some(url))?;

    let block_spec = BlockSpec::Number(CANYON_BLOCK_NUMBER);
    let (_, hardfork) = fixture
        .provider_data
        .create_evm_config_at_block_spec(&block_spec)?;

    assert_eq!(hardfork, OptimismSpecId::CANYON);

    let chain_id = fixture.provider_data.chain_id_at_block_spec(&block_spec)?;
    assert_eq!(chain_id, SEPOLIA_CHAIN_ID);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sepolia_call_with_remote_chain_id() -> anyhow::Result<()> {
    const GAS_PRICE_ORACLE_L1_BLOCK_ADDRESS: Address =
        address!("420000000000000000000000000000000000000F");

    let logger = Box::new(NoopLogger::<OptimismChainSpec>::default());
    let subscriber = Box::new(|_event| {});

    let mut config = create_test_config_with_fork(Some(ForkConfig {
        json_rpc_url: sepolia_url(),
        block_number: None,
        http_headers: None,
    }));

    // Set a different chain ID than the forked chain ID
    config.chain_id = 31337;

    let provider = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        config,
        Arc::<ContractDecoder>::default(),
        CurrentTime,
    )?;

    let last_block_number = {
        let response =
            provider.handle_request(ProviderRequest::Single(MethodInvocation::BlockNumber(())))?;

        serde_json::from_value::<U64>(response.result)?.to::<u64>()
    };

    let data = bytes!("de26c4a10000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000002c02ea827a6981c4843b9aca00843b9c24e382520994f39fd6e51aad88f6f4ce6ab8827279cfffb922660180c00000000000000000000000000000000000000000");
    let _response = provider.handle_request(ProviderRequest::Single(MethodInvocation::Call(
        CallRequest {
            from: Some(address!("f39fd6e51aad88f6f4ce6ab8827279cfffb92266")),
            to: Some(GAS_PRICE_ORACLE_L1_BLOCK_ADDRESS),
            data: Some(data),
            ..CallRequest::default()
        },
        Some(BlockSpec::Number(last_block_number)),
        None,
    )))?;

    Ok(())
}
