#![cfg(feature = "test-remote")]

use std::sync::Arc;

use anyhow::anyhow;
use edr_defaults::CACHE_DIR;
use edr_eth::{b256, transaction::TransactionType as _, HashMap, PreEip1898BlockSpec, B256};
use edr_evm::{
    blockchain::ForkedBlockchain, state::IrregularState, Block, RandomHashGenerator, RemoteBlock,
};
use edr_optimism::{hardfork, transaction, OptimismChainSpec};
use edr_rpc_eth::client::EthRpcClient;
use edr_test_utils::env::get_alchemy_url;
use op_alloy_rpc_types::receipt::L1BlockInfo;
use tokio::runtime;

#[tokio::test(flavor = "multi_thread")]
async fn block_with_transactions() -> anyhow::Result<()> {
    const BLOCK_NUMBER_WITH_TRANSACTIONS: u64 = 117_156_000;

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let block = rpc_client
        .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::Number(
            BLOCK_NUMBER_WITH_TRANSACTIONS,
        ))
        .await?;

    let _block = RemoteBlock::new(block, Arc::new(rpc_client), runtime::Handle::current())?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn block_with_deposit_transaction() -> anyhow::Result<()> {
    const BLOCK_NUMBER_WITH_DEPOSIT: u64 = 121_874_088;
    const CHAIN_ID: u64 = 10;

    let runtime = tokio::runtime::Handle::current();

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;
    let rpc_client = Arc::new(rpc_client);

    let replay_block = {
        let block = rpc_client
            .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::Number(
                BLOCK_NUMBER_WITH_DEPOSIT,
            ))
            .await?;

        RemoteBlock::new(block, rpc_client.clone(), runtime.clone())?
    };

    let mut irregular_state = IrregularState::default();
    let state_root_generator = Arc::new(parking_lot::Mutex::new(RandomHashGenerator::with_seed(
        edr_defaults::STATE_ROOT_HASH_SEED,
    )));
    let hardfork_activation_overrides = HashMap::new();

    let hardfork_activations =
        hardfork::chain_hardfork_activations(CHAIN_ID).ok_or(anyhow!("Unsupported chain id"))?;

    let spec_id = hardfork_activations
        .hardfork_at_block(BLOCK_NUMBER_WITH_DEPOSIT, replay_block.header().timestamp)
        .ok_or(anyhow!("Unsupported block"))?;

    let _blockchain = ForkedBlockchain::new(
        runtime.clone(),
        None,
        spec_id,
        rpc_client.clone(),
        Some(BLOCK_NUMBER_WITH_DEPOSIT - 1),
        &mut irregular_state,
        state_root_generator,
        &hardfork_activation_overrides,
    )
    .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn transaction_and_receipt_pre_bedrock() -> anyhow::Result<()> {
    const TRANSACTION_HASH: B256 =
        b256!("29a264d922bcf76ebae7a60af9ca405ff8946fde819123b1beafca43f5753ce1");

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let transaction = rpc_client
        .get_transaction_by_hash(TRANSACTION_HASH)
        .await?
        .expect("Transaction must exist");

    let transaction = transaction::Signed::try_from(transaction)?;
    assert_eq!(transaction.transaction_type(), transaction::Type::Legacy);

    let receipt = rpc_client
        .get_transaction_receipt(TRANSACTION_HASH)
        .await?
        .expect("Receipt must exist");

    assert_eq!(receipt.transaction_type, None);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn deposit_transaction_and_receipt_regolith() -> anyhow::Result<()> {
    const TRANSACTION_HASH: B256 =
        b256!("dd8e089476419b44cc37d72e631c44c57b38ac5a25fe5dea7b38688b83022fa1");

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let transaction = rpc_client
        .get_transaction_by_hash(TRANSACTION_HASH)
        .await?
        .expect("Transaction must exist");

    let transaction = transaction::Signed::try_from(transaction)?;
    assert_eq!(transaction.transaction_type(), transaction::Type::Deposit);

    let receipt = rpc_client
        .get_transaction_receipt(TRANSACTION_HASH)
        .await?
        .expect("Receipt must exist");

    assert_eq!(
        receipt.transaction_type,
        Some(transaction::Type::Deposit.into())
    );
    assert!(receipt.deposit_receipt_version.is_none());
    assert_l1_block_info_is_none(&receipt.l1_block_info);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn deposit_transaction_and_receipt_canyon() -> anyhow::Result<()> {
    const TRANSACTION_HASH: B256 =
        b256!("64c32c8d474e8befdea12e25338ad86d53950b1156c413f409e785112cfed4d3");

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let transaction = rpc_client
        .get_transaction_by_hash(TRANSACTION_HASH)
        .await?
        .expect("Transaction must exist");

    let transaction = transaction::Signed::try_from(transaction)?;
    assert_eq!(transaction.transaction_type(), transaction::Type::Deposit);

    let receipt = rpc_client
        .get_transaction_receipt(TRANSACTION_HASH)
        .await?
        .expect("Receipt must exist");

    assert_eq!(
        receipt.transaction_type,
        Some(transaction::Type::Deposit.into())
    );
    assert_eq!(receipt.deposit_receipt_version, Some(1));
    assert_l1_block_info_is_none(&receipt.l1_block_info);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn deposit_transaction_and_receipt_ecotone() -> anyhow::Result<()> {
    const TRANSACTION_HASH: B256 =
        b256!("cca2f31992022e3a833959c505de021285a7c5339c8d1b8ad75100074e1c6aea");

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let transaction = rpc_client
        .get_transaction_by_hash(TRANSACTION_HASH)
        .await?
        .expect("Transaction must exist");

    let transaction = transaction::Signed::try_from(transaction)?;
    assert_eq!(transaction.transaction_type(), transaction::Type::Deposit);

    let receipt = rpc_client
        .get_transaction_receipt(TRANSACTION_HASH)
        .await?
        .expect("Receipt must exist");

    assert_eq!(
        receipt.transaction_type,
        Some(transaction::Type::Deposit.into())
    );
    assert_eq!(receipt.deposit_receipt_version, Some(1));
    assert_l1_block_info_is_none(&receipt.l1_block_info);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn receipt_with_l1_block_info() -> anyhow::Result<()> {
    const TRANSACTION_HASH: B256 =
        b256!("f0b04a1c6f61b2818ac2c62ed0c3fc22cd7ebd2f51161759714f75dd27fa7caa");

    let url = get_alchemy_url().replace("eth-", "opt-");
    let rpc_client = EthRpcClient::<OptimismChainSpec>::new(&url, CACHE_DIR.into(), None)?;

    let receipt = rpc_client
        .get_transaction_receipt(TRANSACTION_HASH)
        .await?
        .expect("Receipt must exist");

    assert_eq!(receipt.l1_block_info.l1_gas_price, Some(0x5f3a77dd6));
    assert_eq!(receipt.l1_block_info.l1_gas_used, Some(0x640));
    assert_eq!(receipt.l1_block_info.l1_fee, Some(0x1c3441e5e02));
    assert_eq!(receipt.l1_block_info.l1_fee_scalar, None);
    assert_eq!(receipt.l1_block_info.l1_base_fee_scalar, Some(0x146b));
    assert_eq!(receipt.l1_block_info.l1_blob_base_fee, Some(0x3f5694c1f));
    assert_eq!(receipt.l1_block_info.l1_blob_base_fee_scalar, Some(0xf79c5));

    Ok(())
}

fn assert_l1_block_info_is_none(l1_block_info: &L1BlockInfo) {
    assert!(l1_block_info.l1_gas_price.is_none());
    assert!(l1_block_info.l1_gas_used.is_none());
    assert!(l1_block_info.l1_fee.is_none());
    assert!(l1_block_info.l1_fee_scalar.is_none());
    assert!(l1_block_info.l1_base_fee_scalar.is_none());
    assert!(l1_block_info.l1_blob_base_fee.is_none());
    assert!(l1_block_info.l1_blob_base_fee_scalar.is_none());
}
