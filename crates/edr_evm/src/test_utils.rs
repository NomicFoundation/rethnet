use std::{fmt::Debug, num::NonZeroU64, sync::Arc};

use anyhow::anyhow;
use edr_eth::{
    account::AccountInfo,
    block::{miner_reward, BlockOptions},
    l1::{self, L1ChainSpec},
    log::FilterLog,
    receipt::{AsExecutionReceipt, ExecutionReceipt as _, ReceiptTrait as _},
    result::InvalidTransaction,
    transaction::{TransactionValidation, TxKind},
    withdrawal::Withdrawal,
    Address, Bytes, HashMap, PreEip1898BlockSpec, U256,
};
use edr_rpc_eth::client::EthRpcClient;

use crate::{
    blockchain::{Blockchain as _, BlockchainErrorForChainSpec, ForkedBlockchain},
    config::CfgEnv,
    spec::SyncRuntimeSpec,
    state::{AccountTrie, IrregularState, StateError, TrieState},
    transaction, Block, BlockBuilder, BlockReceipts, LocalBlock as _, MemPool,
    MemPoolAddTransactionError, RandomHashGenerator, RemoteBlock,
};

/// A test fixture for `MemPool`.
pub struct MemPoolTestFixture {
    /// The mem pool.
    pub mem_pool: MemPool<L1ChainSpec>,
    /// The state.
    pub state: TrieState,
}

impl MemPoolTestFixture {
    /// Constructs an instance with the provided accounts.
    pub fn with_accounts(accounts: &[(Address, AccountInfo)]) -> Self {
        let accounts = accounts.iter().cloned().collect::<HashMap<_, _>>();
        let trie = AccountTrie::with_accounts(&accounts);

        MemPoolTestFixture {
            // SAFETY: literal is non-zero
            mem_pool: MemPool::new(unsafe { NonZeroU64::new_unchecked(10_000_000u64) }),
            state: TrieState::with_accounts(trie),
        }
    }

    /// Tries to add the provided transaction to the mem pool.
    pub fn add_transaction(
        &mut self,
        transaction: transaction::Signed,
    ) -> Result<(), MemPoolAddTransactionError<StateError>> {
        self.mem_pool.add_transaction(&self.state, transaction)
    }

    /// Sets the block gas limit.
    pub fn set_block_gas_limit(&mut self, block_gas_limit: NonZeroU64) -> Result<(), StateError> {
        self.mem_pool
            .set_block_gas_limit(&self.state, block_gas_limit)
    }

    /// Updates the mem pool.
    pub fn update(&mut self) -> Result<(), StateError> {
        self.mem_pool.update(&self.state)
    }
}

/// Creates a dummy EIP-155 transaction.
pub fn dummy_eip155_transaction(
    caller: Address,
    nonce: u64,
) -> Result<transaction::Signed, transaction::CreationError> {
    dummy_eip155_transaction_with_price(caller, nonce, U256::ZERO)
}

/// Creates a dummy EIP-155 transaction with the provided gas price.
pub fn dummy_eip155_transaction_with_price(
    caller: Address,
    nonce: u64,
    gas_price: U256,
) -> Result<transaction::Signed, transaction::CreationError> {
    dummy_eip155_transaction_with_price_and_limit(caller, nonce, gas_price, 30_000)
}

/// Creates a dummy EIP-155 transaction with the provided gas limit.
pub fn dummy_eip155_transaction_with_limit(
    caller: Address,
    nonce: u64,
    gas_limit: u64,
) -> Result<transaction::Signed, transaction::CreationError> {
    dummy_eip155_transaction_with_price_and_limit(caller, nonce, U256::ZERO, gas_limit)
}

fn dummy_eip155_transaction_with_price_and_limit(
    caller: Address,
    nonce: u64,
    gas_price: U256,
    gas_limit: u64,
) -> Result<transaction::Signed, transaction::CreationError> {
    dummy_eip155_transaction_with_price_limit_and_value(
        caller,
        nonce,
        gas_price,
        gas_limit,
        U256::ZERO,
    )
}

/// Creates a dummy EIP-155 transaction with the provided gas price, gas limit,
/// and value.
pub fn dummy_eip155_transaction_with_price_limit_and_value(
    caller: Address,
    nonce: u64,
    gas_price: U256,
    gas_limit: u64,
    value: U256,
) -> Result<transaction::Signed, transaction::CreationError> {
    let from = Address::random();
    let request = transaction::request::Eip155 {
        nonce,
        gas_price,
        gas_limit,
        kind: TxKind::Call(from),
        value,
        input: Bytes::new(),
        chain_id: 123,
    };
    let transaction = request.fake_sign(caller);
    let transaction = transaction::Signed::from(transaction);

    transaction::validate(transaction, l1::SpecId::LATEST)
}

/// Creates a dummy EIP-1559 transaction with the provided max fee and max
/// priority fee per gas.
pub fn dummy_eip1559_transaction(
    caller: Address,
    nonce: u64,
    max_fee_per_gas: U256,
    max_priority_fee_per_gas: U256,
) -> Result<transaction::Signed, transaction::CreationError> {
    let from = Address::random();
    let request = transaction::request::Eip1559 {
        chain_id: 123,
        nonce,
        max_priority_fee_per_gas,
        max_fee_per_gas,
        gas_limit: 30_000,
        kind: TxKind::Call(from),
        value: U256::ZERO,
        input: Bytes::new(),
        access_list: Vec::new(),
    };
    let transaction = request.fake_sign(caller);
    let transaction = transaction::Signed::from(transaction);

    transaction::validate(transaction, l1::SpecId::LATEST)
}

/// Runs a full remote block, asserting that the mined block matches the remote
/// block.
pub async fn run_full_block<
    ChainSpecT: Debug
        + SyncRuntimeSpec<
            BlockEnv: Default,
            BlockReceipt: AsExecutionReceipt<
                ExecutionReceipt = ChainSpecT::ExecutionReceipt<FilterLog>,
            >,
            ExecutionReceipt<FilterLog>: PartialEq,
            LocalBlock: BlockReceipts<
                Arc<ChainSpecT::BlockReceipt>,
                Error = BlockchainErrorForChainSpec<ChainSpecT>,
            >,
            SignedTransaction: Default
                                   + TransactionValidation<
                ValidationError: From<InvalidTransaction> + Send + Sync,
            >,
        >,
>(
    url: String,
    block_number: u64,
) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Handle::current();

    let rpc_client = EthRpcClient::<ChainSpecT>::new(&url, edr_defaults::CACHE_DIR.into(), None)?;
    let chain_id = rpc_client.chain_id().await?;

    let rpc_client = Arc::new(rpc_client);
    let replay_block = {
        let block = rpc_client
            .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::Number(block_number))
            .await?;

        RemoteBlock::new(block, rpc_client.clone(), runtime.clone())?
    };

    let mut irregular_state = IrregularState::default();
    let state_root_generator = Arc::new(parking_lot::Mutex::new(RandomHashGenerator::with_seed(
        edr_defaults::STATE_ROOT_HASH_SEED,
    )));
    let hardfork_activation_overrides = HashMap::new();

    let hardfork_activations =
        ChainSpecT::chain_hardfork_activations(chain_id).ok_or(anyhow!("Unsupported chain id"))?;

    let replay_header = replay_block.header();
    let hardfork = hardfork_activations
        .hardfork_at_block(block_number, replay_header.timestamp)
        .ok_or(anyhow!("Unsupported block number"))?;

    let blockchain = ForkedBlockchain::new(
        runtime.clone(),
        Some(chain_id),
        hardfork,
        rpc_client,
        Some(block_number - 1),
        &mut irregular_state,
        state_root_generator,
        &hardfork_activation_overrides,
    )
    .await?;

    let mut cfg = CfgEnv::default();
    cfg.chain_id = chain_id;
    cfg.disable_eip3607 = true;

    let state =
        blockchain.state_at_block_number(block_number - 1, irregular_state.state_overrides())?;

    let mut builder = ChainSpecT::BlockBuilder::<'_, _, (), _>::new_block_builder(
        &blockchain,
        state,
        hardfork,
        cfg,
        BlockOptions {
            beneficiary: Some(replay_header.beneficiary),
            gas_limit: Some(replay_header.gas_limit),
            extra_data: Some(replay_header.extra_data.clone()),
            mix_hash: Some(replay_header.mix_hash),
            nonce: Some(replay_header.nonce),
            parent_beacon_block_root: replay_header.parent_beacon_block_root,
            state_root: Some(replay_header.state_root),
            timestamp: Some(replay_header.timestamp),
            withdrawals: replay_block.withdrawals().map(<[Withdrawal]>::to_vec),
            ..BlockOptions::default()
        },
        None,
    )?;

    assert_eq!(replay_header.base_fee_per_gas, builder.header().base_fee);

    for transaction in replay_block.transactions() {
        builder = builder
            .add_transaction(transaction.clone())
            .map_err(|error| error.error)?;
    }

    let rewards = vec![(
        replay_header.beneficiary,
        miner_reward(hardfork.into()).unwrap_or(U256::ZERO),
    )];
    let mined_block = builder.finalize(rewards)?;

    let mined_header = mined_block.block.header();
    for (expected, actual) in replay_block
        .fetch_transaction_receipts()?
        .into_iter()
        .zip(mined_block.block.transaction_receipts().iter())
    {
        debug_assert_eq!(
            expected.block_number(),
            actual.block_number(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.transaction_hash(),
            actual.transaction_hash(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.transaction_index(),
            actual.transaction_index(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.from(),
            actual.from(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.to(),
            actual.to(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.contract_address(),
            actual.contract_address(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.gas_used(),
            actual.gas_used(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        // Skip effective gas price check because Hardhat doesn't include it pre-London
        // debug_assert_eq!(
        //     expected.effective_gas_price,
        //     actual.effective_gas_price,
        //     "{:?}",
        //     replay_block.transactions()[expected.transaction_index as usize]
        // );
        debug_assert_eq!(
            expected.cumulative_gas_used(),
            actual.cumulative_gas_used(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        if expected.logs_bloom() != actual.logs_bloom() {
            for (expected, actual) in expected
                .transaction_logs()
                .iter()
                .zip(actual.transaction_logs().iter())
            {
                debug_assert_eq!(
                    expected.inner.address,
                    actual.inner.address,
                    "{:?}",
                    replay_block.transactions()[expected.transaction_index as usize]
                );
                debug_assert_eq!(
                    expected.inner.topics(),
                    actual.inner.topics(),
                    "{:?}",
                    replay_block.transactions()[expected.transaction_index as usize]
                );
                debug_assert_eq!(
                    expected.inner.data.data,
                    actual.inner.data.data,
                    "{:?}",
                    replay_block.transactions()[expected.transaction_index as usize]
                );
            }
        }
        debug_assert_eq!(
            expected.root_or_status(),
            actual.root_or_status(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
        debug_assert_eq!(
            expected.as_execution_receipt(),
            actual.as_execution_receipt(),
            "{:?}",
            replay_block.transactions()[expected.transaction_index() as usize]
        );
    }

    assert_eq!(replay_header, mined_header);

    Ok(())
}

/// Implements full block tests for the provided chain specs.
/// ```no_run
/// use edr_eth::l1::L1ChainSpec;
/// use edr_evm::impl_full_block_tests;
/// use edr_test_utils::env::get_alchemy_url;
///
/// impl_full_block_tests! {
///     mainnet_byzantium => L1ChainSpec {
///         block_number: 4_370_001,
///         url: get_alchemy_url(),
///     },
/// }
/// ```
#[macro_export]
macro_rules! impl_full_block_tests {
    ($(
        $name:ident => $chain_spec:ident {
            block_number: $block_number:expr,
            url: $url:expr,
        },
    )+) => {
        $(
            paste::item! {
                #[serial_test::serial]
                #[tokio::test(flavor = "multi_thread")]
                async fn [<full_block_ $name>]() -> anyhow::Result<()> {
                    let url = $url;

                    $crate::test_utils::run_full_block::<$chain_spec>(url, $block_number).await
                }
            }
        )+
    }
}
