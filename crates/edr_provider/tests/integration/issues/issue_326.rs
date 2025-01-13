use std::{str::FromStr, sync::Arc};

use edr_eth::{
    account::AccountInfo,
    l1::{self, L1ChainSpec},
    Address, KECCAK_EMPTY, U256,
};
use edr_provider::{
    test_utils::{create_test_config_with_fork, one_ether},
    time::CurrentTime,
    MethodInvocation, MiningConfig, NoopLogger, Provider, ProviderRequest,
};
use edr_rpc_eth::{CallRequest, TransactionRequest};
use edr_solidity::contract_decoder::ContractDecoder;
use tokio::runtime;

#[tokio::test(flavor = "multi_thread")]
async fn issue_326() -> anyhow::Result<()> {
    let logger = Box::new(NoopLogger::<L1ChainSpec>::default());
    let subscriber = Box::new(|_event| {});

    let mut config = create_test_config_with_fork(None);
    config.hardfork = l1::SpecId::CANCUN;
    config.mining = MiningConfig {
        auto_mine: false,
        ..MiningConfig::default()
    };
    config.initial_base_fee_per_gas = Some(U256::from_str("0x100").unwrap());

    let impersonated_account = Address::random();
    config.genesis_state.insert(
        impersonated_account,
        AccountInfo {
            balance: one_ether(),
            nonce: 0,
            code: None,
            code_hash: KECCAK_EMPTY,
        }
        .into(),
    );

    let provider = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        config,
        Arc::<ContractDecoder>::default(),
        CurrentTime,
    )?;

    provider.handle_request(ProviderRequest::Single(
        MethodInvocation::ImpersonateAccount(impersonated_account.into()),
    ))?;

    provider.handle_request(ProviderRequest::Single(MethodInvocation::Mine(None, None)))?;

    provider.handle_request(ProviderRequest::Single(MethodInvocation::SendTransaction(
        TransactionRequest {
            from: impersonated_account,
            to: Some(impersonated_account),
            nonce: Some(0),
            max_fee_per_gas: Some(U256::from_str("0xA").unwrap()),
            ..TransactionRequest::default()
        },
    )))?;

    provider.handle_request(ProviderRequest::Single(MethodInvocation::EstimateGas(
        CallRequest {
            from: Some(impersonated_account),
            to: Some(impersonated_account),
            max_fee_per_gas: Some(U256::from_str("0x200").unwrap()),
            ..CallRequest::default()
        },
        None,
    )))?;

    Ok(())
}
