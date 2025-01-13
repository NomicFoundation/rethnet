use core::fmt::Debug;
use std::{num::NonZeroU64, sync::Arc, time::SystemTime};

use anyhow::anyhow;
use edr_eth::{
    account::AccountInfo,
    block::BlobGas,
    l1::L1ChainSpec,
    result::InvalidTransaction,
    signature::secret_key_from_str,
    spec::HardforkTrait,
    transaction::{self, request::TransactionRequestAndSender, TransactionValidation, TxKind},
    trie::KECCAK_NULL_RLP,
    Address, Bytes, HashMap, B256, KECCAK_EMPTY, U160, U256,
};
use edr_evm::Block as _;
use edr_rpc_eth::TransactionRequest;
use edr_solidity::contract_decoder::ContractDecoder;
use tokio::runtime;

use crate::{
    config,
    requests::hardhat::rpc_types::ForkConfig,
    time::{CurrentTime, TimeSinceEpoch},
    MethodInvocation, NoopLogger, Provider, ProviderConfig, ProviderData, ProviderError,
    ProviderRequest, ProviderSpec, SyncProviderSpec,
};

pub const TEST_SECRET_KEY: &str =
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

// Address 0xCD2a3d9F938E13CD947Ec05AbC7FE734Df8DD826
pub const TEST_SECRET_KEY_SIGN_TYPED_DATA_V4: &str =
    "0xc85ef7d79691fe79573b1a7064c19c1a9819ebdbd1faaab1a8ec92344438aaf4";

pub const FORK_BLOCK_NUMBER: u64 = 18_725_000;

/// Constructs a test config with a single account with 1 ether
pub fn create_test_config<HardforkT: HardforkTrait>() -> ProviderConfig<HardforkT> {
    create_test_config_with_fork(None)
}

pub fn one_ether() -> U256 {
    U256::from(10).pow(U256::from(18))
}

pub fn create_test_config_with_fork<HardforkT: HardforkTrait>(
    fork: Option<ForkConfig>,
) -> ProviderConfig<HardforkT> {
    ProviderConfig {
        accounts: vec![
            config::OwnedAccount {
                secret_key: secret_key_from_str(TEST_SECRET_KEY)
                    .expect("should construct secret key from string"),
                balance: one_ether(),
            },
            config::OwnedAccount {
                secret_key: secret_key_from_str(TEST_SECRET_KEY_SIGN_TYPED_DATA_V4)
                    .expect("should construct secret key from string"),
                balance: one_ether(),
            },
        ],
        allow_blocks_with_same_timestamp: false,
        allow_unlimited_contract_size: false,
        bail_on_call_failure: false,
        bail_on_transaction_failure: false,
        // SAFETY: literal is non-zero
        block_gas_limit: unsafe { NonZeroU64::new_unchecked(30_000_000) },
        chain_id: 123,
        chains: HashMap::new(),
        coinbase: Address::from(U160::from(1)),
        enable_rip_7212: false,
        fork,
        genesis_state: HashMap::new(),
        hardfork: HardforkT::default(),
        initial_base_fee_per_gas: Some(U256::from(1000000000)),
        initial_blob_gas: Some(BlobGas {
            gas_used: 0,
            excess_gas: 0,
        }),
        initial_date: Some(SystemTime::now()),
        initial_parent_beacon_block_root: Some(KECCAK_NULL_RLP),
        min_gas_price: U256::ZERO,
        mining: config::Mining::default(),
        network_id: 123,
        cache_dir: edr_defaults::CACHE_DIR.into(),
    }
}

/// Retrieves the pending base fee per gas from the provider data.
pub fn pending_base_fee<
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    let block = data.mine_pending_block()?.block;

    let base_fee = block
        .header()
        .base_fee_per_gas
        .unwrap_or_else(|| U256::from(1));

    Ok(base_fee)
}

/// Deploys a contract with the provided code. Returns the address of the
/// contract.
pub fn deploy_contract<TimerT>(
    provider: &Provider<L1ChainSpec, TimerT>,
    caller: Address,
    code: Bytes,
) -> anyhow::Result<Address>
where
    TimerT: Clone + TimeSinceEpoch,
{
    let deploy_transaction = TransactionRequest {
        from: caller,
        data: Some(code),
        ..TransactionRequest::default()
    };

    let result = provider.handle_request(ProviderRequest::Single(
        MethodInvocation::SendTransaction(deploy_transaction),
    ))?;

    let transaction_hash: B256 = serde_json::from_value(result.result)?;

    let result = provider.handle_request(ProviderRequest::Single(
        MethodInvocation::GetTransactionReceipt(transaction_hash),
    ))?;

    let receipt: edr_rpc_eth::receipt::Block = serde_json::from_value(result.result)?;
    let contract_address = receipt.contract_address.expect("Call must create contract");

    Ok(contract_address)
}

/// Fixture for testing `ProviderData`.
pub struct ProviderTestFixture<ChainSpecT: ProviderSpec<CurrentTime>> {
    _runtime: runtime::Runtime,
    pub config: ProviderConfig<ChainSpecT::Hardfork>,
    pub provider_data: ProviderData<ChainSpecT, CurrentTime>,
    pub impersonated_account: Address,
}

impl<ChainSpecT: Debug + SyncProviderSpec<CurrentTime>> ProviderTestFixture<ChainSpecT> {
    /// Creates a new `ProviderTestFixture` with a local provider.
    pub fn new_local() -> anyhow::Result<Self> {
        Self::with_fork(None)
    }

    /// Creates a new `ProviderTestFixture` with a forked provider.
    pub fn new_forked(url: Option<String>) -> anyhow::Result<Self> {
        use edr_test_utils::env::get_alchemy_url;

        let fork_url = url.unwrap_or(get_alchemy_url());
        Self::with_fork(Some(fork_url))
    }

    fn with_fork(fork: Option<String>) -> anyhow::Result<Self> {
        let fork = fork.map(|json_rpc_url| {
            ForkConfig {
                json_rpc_url,
                // Random recent block for better cache consistency
                block_number: Some(FORK_BLOCK_NUMBER),
                http_headers: None,
            }
        });

        let config = create_test_config_with_fork(fork);

        let runtime = runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("provider-data-test")
            .build()?;

        Self::new(runtime, config)
    }

    pub fn new(
        runtime: tokio::runtime::Runtime,
        mut config: ProviderConfig<ChainSpecT::Hardfork>,
    ) -> anyhow::Result<Self> {
        let logger = Box::<NoopLogger<ChainSpecT>>::default();
        let subscription_callback_noop = Box::new(|_| ());

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

        let mut provider_data = ProviderData::<ChainSpecT>::new(
            runtime.handle().clone(),
            logger,
            subscription_callback_noop,
            None,
            config.clone(),
            Arc::<ContractDecoder>::default(),
            CurrentTime,
        )?;

        provider_data.impersonate_account(impersonated_account);

        Ok(Self {
            _runtime: runtime,
            config,
            provider_data,
            impersonated_account,
        })
    }

    /// Retrieves the nth local account.
    ///
    /// # Panics
    ///
    /// Panics if there are not enough local accounts
    pub fn nth_local_account(&self, index: usize) -> anyhow::Result<Address> {
        self.provider_data
            .accounts()
            .nth(index)
            .copied()
            .ok_or(anyhow!("the requested local account does not exist"))
    }
}

impl ProviderTestFixture<L1ChainSpec> {
    pub fn dummy_transaction_request(
        &self,
        local_account_index: usize,
        gas_limit: u64,
        nonce: Option<u64>,
    ) -> anyhow::Result<TransactionRequestAndSender<transaction::Request>> {
        let request = transaction::Request::Eip155(transaction::request::Eip155 {
            kind: TxKind::Call(Address::ZERO),
            gas_limit,
            gas_price: U256::from(42_000_000_000_u64),
            value: U256::from(1),
            input: Bytes::default(),
            nonce: nonce.unwrap_or(0),
            chain_id: self.config.chain_id,
        });

        let sender = self.nth_local_account(local_account_index)?;
        Ok(TransactionRequestAndSender { request, sender })
    }

    pub fn impersonated_dummy_transaction(&self) -> anyhow::Result<transaction::Signed> {
        let mut transaction = self.dummy_transaction_request(0, 30_000, None)?;
        transaction.sender = self.impersonated_account;

        Ok(self.provider_data.sign_transaction_request(transaction)?)
    }

    pub fn signed_dummy_transaction(
        &self,
        local_account_index: usize,
        nonce: Option<u64>,
    ) -> anyhow::Result<transaction::Signed> {
        let transaction = self.dummy_transaction_request(local_account_index, 30_000, nonce)?;
        Ok(self.provider_data.sign_transaction_request(transaction)?)
    }
}
