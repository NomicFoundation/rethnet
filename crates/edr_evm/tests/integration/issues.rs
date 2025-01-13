#![cfg(feature = "test-remote")]

use std::{str::FromStr, sync::Arc};

use edr_defaults::CACHE_DIR;
use edr_eth::{
    l1::{self, L1ChainSpec},
    Address, HashMap, U256,
};
use edr_evm::{
    blockchain::ForkedBlockchain,
    precompile::{self, Precompiles},
    state::{AccountModifierFn, ForkState, IrregularState, StateDebug},
    RandomHashGenerator,
};
use edr_rpc_eth::client::EthRpcClient;
use edr_test_utils::env::get_alchemy_url;
use parking_lot::Mutex;
use tokio::runtime;

#[tokio::test(flavor = "multi_thread")]
async fn issue_336_set_balance_after_forking() -> anyhow::Result<()> {
    const TEST_CONTRACT_ADDRESS: &str = "0x530B7F66914c1E345DF1683eae4536fc7b80660f";
    const DEPLOYMENT_BLOCK_NUMBER: u64 = 5464258;

    let contract_address = Address::from_str(TEST_CONTRACT_ADDRESS).unwrap();

    let rpc_client = EthRpcClient::<L1ChainSpec>::new(
        &get_alchemy_url().replace("mainnet", "sepolia"),
        CACHE_DIR.into(),
        None,
    )?;

    let mut state_root_generator = RandomHashGenerator::with_seed("test");
    let state_root = state_root_generator.generate_next();

    let mut state = ForkState::new(
        runtime::Handle::current(),
        Arc::new(rpc_client),
        Arc::new(Mutex::new(state_root_generator)),
        DEPLOYMENT_BLOCK_NUMBER,
        state_root,
    );

    state.modify_account(
        contract_address,
        AccountModifierFn::new(Box::new(|balance, _nonce, _code| {
            *balance += U256::from(1);
        })),
    )?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn issue_hh_4974_forking_avalanche_c_chain() -> anyhow::Result<()> {
    const FORK_BLOCK_NUMBER: u64 = 12_508_443;

    let url = "https://coston-api.flare.network/ext/bc/C/rpc";
    let rpc_client = EthRpcClient::<L1ChainSpec>::new(url, CACHE_DIR.into(), None)?;
    let mut irregular_state = IrregularState::default();
    let state_root_generator = Arc::new(Mutex::new(RandomHashGenerator::with_seed("test")));
    let hardfork_activation_overrides = HashMap::new();

    let _blockchain = ForkedBlockchain::new(
        runtime::Handle::current(),
        None,
        l1::SpecId::LATEST,
        Arc::new(rpc_client),
        Some(FORK_BLOCK_NUMBER),
        &mut irregular_state,
        state_root_generator,
        &hardfork_activation_overrides,
    )
    .await?;

    Ok(())
}

#[test]
fn issue_364_kzg_point_evaluation_present_in_cancun() {
    const KZG_POINT_EVALUATION_ADDRESS: Address = precompile::u64_to_address(0x0A);

    let precompiles = Precompiles::cancun();
    assert!(precompiles.contains(&KZG_POINT_EVALUATION_ADDRESS));
}
