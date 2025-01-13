use core::fmt::Debug;
use std::sync::Arc;

use edr_eth::{
    result::InvalidTransaction,
    transaction::{ExecutableTransaction as _, TransactionValidation},
    BlockSpec, PreEip1898BlockSpec, B256, U256, U64,
};
use edr_evm::{
    block::transaction::{BlockDataForTransaction, TransactionAndBlock},
    spec::RuntimeSpec,
    Block as _,
};
use edr_rpc_eth::RpcTypeFrom as _;

use crate::{
    data::ProviderData, requests::validation::validate_post_merge_block_tags,
    spec::SyncProviderSpec, time::TimeSinceEpoch, ProviderError,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum HashOrTransaction<ChainSpecT: RuntimeSpec> {
    Hash(B256),
    Transaction(ChainSpecT::RpcTransaction),
}

pub fn handle_get_block_by_hash_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    block_hash: B256,
    transaction_detail_flag: bool,
) -> Result<Option<edr_rpc_eth::Block<HashOrTransaction<ChainSpecT>>>, ProviderError<ChainSpecT>> {
    data.block_by_hash(&block_hash)?
        .map(|block| {
            let total_difficulty = data.total_difficulty_by_hash(block.block_hash())?;
            let pending = false;
            block_to_rpc_output(
                data.hardfork(),
                block,
                pending,
                total_difficulty,
                transaction_detail_flag,
            )
        })
        .transpose()
}

pub fn handle_get_block_by_number_request<
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
    block_spec: PreEip1898BlockSpec,
    transaction_detail_flag: bool,
) -> Result<Option<edr_rpc_eth::Block<HashOrTransaction<ChainSpecT>>>, ProviderError<ChainSpecT>> {
    block_by_number(data, &block_spec.into())?
        .map(
            |BlockByNumberResult {
                 block,
                 pending,
                 total_difficulty,
             }| {
                block_to_rpc_output(
                    data.hardfork(),
                    block,
                    pending,
                    total_difficulty,
                    transaction_detail_flag,
                )
            },
        )
        .transpose()
}

pub fn handle_get_block_transaction_count_by_hash_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    block_hash: B256,
) -> Result<Option<U64>, ProviderError<ChainSpecT>> {
    Ok(data
        .block_by_hash(&block_hash)?
        .map(|block| U64::from(block.transactions().len())))
}

pub fn handle_get_block_transaction_count_by_block_number<
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
    block_spec: PreEip1898BlockSpec,
) -> Result<Option<U64>, ProviderError<ChainSpecT>> {
    Ok(block_by_number(data, &block_spec.into())?
        .map(|BlockByNumberResult { block, .. }| U64::from(block.transactions().len())))
}

/// Helper type for a chain-specific [`BlockByNumberResult`].
type BlockByNumberResultForChainSpec<ChainSpecT> =
    BlockByNumberResult<Arc<<ChainSpecT as RuntimeSpec>::Block>>;

/// The result returned by requesting a block by number.
#[derive(Clone, Debug)]
struct BlockByNumberResult<BlockT> {
    /// The block
    pub block: BlockT,
    /// Whether the block is a pending block.
    pub pending: bool,
    /// The total difficulty with the block
    pub total_difficulty: Option<U256>,
}

fn block_by_number<
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
    block_spec: &BlockSpec,
) -> Result<Option<BlockByNumberResultForChainSpec<ChainSpecT>>, ProviderError<ChainSpecT>> {
    validate_post_merge_block_tags(data.hardfork(), block_spec)?;

    match data.block_by_block_spec(block_spec) {
        Ok(Some(block)) => {
            let total_difficulty = data.total_difficulty_by_hash(block.block_hash())?;
            Ok(Some(BlockByNumberResult {
                block,
                pending: false,
                total_difficulty,
            }))
        }
        // Pending block
        Ok(None) => {
            let result = data.mine_pending_block()?;
            let pending_block = Arc::new(result.block);

            let last_block = data.last_block()?;
            let previous_total_difficulty = data
                .total_difficulty_by_hash(last_block.block_hash())?
                .expect("last block has total difficulty");
            let total_difficulty = previous_total_difficulty + pending_block.header().difficulty;

            Ok(Some(BlockByNumberResult {
                block: ChainSpecT::cast_local_block(pending_block),
                pending: true,
                total_difficulty: Some(total_difficulty),
            }))
        }
        Err(ProviderError::InvalidBlockNumberOrHash { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

fn block_to_rpc_output<ChainSpecT: RuntimeSpec>(
    hardfork: ChainSpecT::Hardfork,
    block: Arc<ChainSpecT::Block>,
    is_pending: bool,
    total_difficulty: Option<U256>,
    transaction_detail_flag: bool,
) -> Result<edr_rpc_eth::Block<HashOrTransaction<ChainSpecT>>, ProviderError<ChainSpecT>> {
    let header = block.header();

    let transactions: Vec<HashOrTransaction<ChainSpecT>> = if transaction_detail_flag {
        block
            .transactions()
            .iter()
            .enumerate()
            .map(|(i, tx)| TransactionAndBlock {
                transaction: tx.clone(),
                block_data: Some(BlockDataForTransaction {
                    block: block.clone(),
                    transaction_index: i.try_into().expect("usize fits into u64"),
                }),
                is_pending,
            })
            .map(
                |transaction_and_block: TransactionAndBlock<
                    Arc<ChainSpecT::Block>,
                    ChainSpecT::SignedTransaction,
                >| {
                    ChainSpecT::RpcTransaction::rpc_type_from(&transaction_and_block, hardfork)
                },
            )
            .map(HashOrTransaction::Transaction)
            .collect()
    } else {
        block
            .transactions()
            .iter()
            .map(|tx| HashOrTransaction::Hash(*tx.transaction_hash()))
            .collect()
    };

    let mix_hash = if is_pending {
        None
    } else {
        Some(header.mix_hash)
    };
    let nonce = if is_pending { None } else { Some(header.nonce) };
    let number = if is_pending {
        None
    } else {
        Some(header.number)
    };

    Ok(edr_rpc_eth::Block {
        hash: Some(*block.block_hash()),
        parent_hash: header.parent_hash,
        sha3_uncles: header.ommers_hash,
        state_root: header.state_root,
        transactions_root: header.transactions_root,
        receipts_root: header.receipts_root,
        number,
        gas_used: header.gas_used,
        gas_limit: header.gas_limit,
        extra_data: header.extra_data.clone(),
        logs_bloom: header.logs_bloom,
        timestamp: header.timestamp,
        difficulty: header.difficulty,
        total_difficulty,
        uncles: block.ommer_hashes().to_vec(),
        transactions,
        size: block.rlp_size(),
        mix_hash,
        nonce,
        base_fee_per_gas: header.base_fee_per_gas,
        miner: Some(header.beneficiary),
        withdrawals: block
            .withdrawals()
            .map(<[edr_eth::withdrawal::Withdrawal]>::to_vec),
        withdrawals_root: header.withdrawals_root,
        blob_gas_used: header.blob_gas.as_ref().map(|bg| bg.gas_used),
        excess_blob_gas: header.blob_gas.as_ref().map(|bg| bg.excess_gas),
        parent_beacon_block_root: header.parent_beacon_block_root,
    })
}
