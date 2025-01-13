use core::cmp;
use std::sync::Arc;

use edr_eth::{
    block::Header,
    receipt::ReceiptTrait as _,
    result::{ExecutionResult, InvalidTransaction},
    reward_percentile::RewardPercentile,
    transaction::{Transaction as _, TransactionMut, TransactionValidation},
    Address, HashMap, U256,
};
use edr_evm::{
    blockchain::{BlockchainErrorForChainSpec, SyncBlockchain},
    config::CfgEnv,
    precompile::Precompile,
    spec::SyncRuntimeSpec,
    state::{StateError, StateOverrides, SyncState},
    trace::{register_trace_collector_handles, TraceCollector},
    Block as _, BlockReceipts, DebugContext,
};
use itertools::Itertools;

use crate::{
    data::call::{self, RunCallArgs},
    ProviderError,
};

pub(super) struct CheckGasLimitArgs<'a, ChainSpecT: SyncRuntimeSpec> {
    pub blockchain:
        &'a dyn SyncBlockchain<ChainSpecT, BlockchainErrorForChainSpec<ChainSpecT>, StateError>,
    pub header: &'a Header,
    pub state: &'a dyn SyncState<StateError>,
    pub state_overrides: &'a StateOverrides,
    pub cfg_env: CfgEnv,
    pub hardfork: ChainSpecT::Hardfork,
    pub transaction: ChainSpecT::SignedTransaction,
    pub gas_limit: u64,
    pub precompiles: &'a HashMap<Address, Precompile>,
    pub trace_collector: &'a mut TraceCollector<ChainSpecT::HaltReason>,
}

/// Test if the transaction successfully executes with the given gas limit.
/// Returns true on success and return false if the transaction runs out of gas
/// or funds or reverts. Returns an error for any other halt reason.
pub(super) fn check_gas_limit<ChainSpecT>(
    args: CheckGasLimitArgs<'_, ChainSpecT>,
) -> Result<bool, ProviderError<ChainSpecT>>
where
    ChainSpecT: SyncRuntimeSpec<
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionMut
                               + TransactionValidation<ValidationError: From<InvalidTransaction>>,
    >,
{
    let CheckGasLimitArgs {
        blockchain,
        header,
        state,
        state_overrides,
        cfg_env,
        hardfork,
        mut transaction,
        gas_limit,
        precompiles,
        trace_collector,
    } = args;

    transaction.set_gas_limit(gas_limit);

    let result = call::run_call(RunCallArgs {
        blockchain,
        header,
        state,
        state_overrides,
        cfg_env,
        hardfork,
        transaction,
        precompiles,
        debug_context: Some(DebugContext {
            data: trace_collector,
            register_handles_fn: register_trace_collector_handles,
        }),
    })?;

    Ok(matches!(result, ExecutionResult::Success { .. }))
}

pub(super) struct BinarySearchEstimationArgs<'a, ChainSpecT: SyncRuntimeSpec> {
    pub blockchain:
        &'a dyn SyncBlockchain<ChainSpecT, BlockchainErrorForChainSpec<ChainSpecT>, StateError>,
    pub header: &'a Header,
    pub state: &'a dyn SyncState<StateError>,
    pub state_overrides: &'a StateOverrides,
    pub cfg_env: CfgEnv,
    pub hardfork: ChainSpecT::Hardfork,
    pub transaction: ChainSpecT::SignedTransaction,
    pub lower_bound: u64,
    pub upper_bound: u64,
    pub precompiles: &'a HashMap<Address, Precompile>,
    pub trace_collector: &'a mut TraceCollector<ChainSpecT::HaltReason>,
}

/// Search for a tight upper bound on the gas limit that will allow the
/// transaction to execute. Matches Hardhat logic, except it's iterative, not
/// recursive.
pub(super) fn binary_search_estimation<ChainSpecT>(
    args: BinarySearchEstimationArgs<'_, ChainSpecT>,
) -> Result<u64, ProviderError<ChainSpecT>>
where
    ChainSpecT: SyncRuntimeSpec<
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionMut
                               + TransactionValidation<ValidationError: From<InvalidTransaction>>,
    >,
{
    const MAX_ITERATIONS: usize = 20;

    let BinarySearchEstimationArgs {
        blockchain,
        header,
        state,
        state_overrides,
        cfg_env,
        hardfork,
        transaction,
        mut lower_bound,
        mut upper_bound,
        precompiles,
        trace_collector,
    } = args;

    let mut i = 0;

    while upper_bound - lower_bound > min_difference(lower_bound) && i < MAX_ITERATIONS {
        let mut mid = lower_bound + (upper_bound - lower_bound) / 2;
        if i == 0 {
            // Start close to the lower bound as it's assumed to be derived from the gas
            // used by the transaction.
            let initial_mid = 3 * lower_bound;
            mid = cmp::min(mid, initial_mid);
        }

        let success = check_gas_limit(CheckGasLimitArgs {
            blockchain,
            header,
            state,
            state_overrides,
            cfg_env: cfg_env.clone(),
            hardfork,
            transaction: transaction.clone(),
            gas_limit: mid,
            precompiles,
            trace_collector,
        })?;

        if success {
            upper_bound = mid;
        } else {
            lower_bound = mid + 1;
        }

        i += 1;
    }

    Ok(upper_bound)
}

// Matches Hardhat
#[inline]
fn min_difference(lower_bound: u64) -> u64 {
    if lower_bound >= 4_000_000 {
        50_000
    } else if lower_bound >= 1_000_000 {
        10_000
    } else if lower_bound >= 100_000 {
        1_000
    } else if lower_bound >= 50_000 {
        500
    } else if lower_bound >= 30_000 {
        300
    } else {
        200
    }
}

/// Compute miner rewards for percentiles.
pub(super) fn compute_rewards<ChainSpecT>(
    block: &ChainSpecT::Block,
    reward_percentiles: &[RewardPercentile],
) -> Result<Vec<U256>, ProviderError<ChainSpecT>>
where
    ChainSpecT: SyncRuntimeSpec<
        Block: BlockReceipts<
            Arc<ChainSpecT::BlockReceipt>,
            Error = BlockchainErrorForChainSpec<ChainSpecT>,
        >,
    >,
{
    if block.transactions().is_empty() {
        return Ok(reward_percentiles.iter().map(|_| U256::ZERO).collect());
    }

    let base_fee_per_gas = block.header().base_fee_per_gas.unwrap_or_default();

    let gas_used_and_effective_reward = block
        .fetch_transaction_receipts()?
        .iter()
        .enumerate()
        .map(|(i, receipt)| {
            let transaction = &block.transactions()[i];

            let gas_used = receipt.gas_used();
            // gas price pre EIP-1559 and max fee per gas post EIP-1559
            let gas_price = transaction.gas_price();

            let effective_reward =
                if let Some(max_priority_fee_per_gas) = transaction.max_priority_fee_per_gas() {
                    cmp::min(*max_priority_fee_per_gas, gas_price - base_fee_per_gas)
                } else {
                    gas_price.saturating_sub(base_fee_per_gas)
                };

            (gas_used, effective_reward)
        })
        .sorted_by(|(_, reward_first), (_, reward_second)| reward_first.cmp(reward_second))
        .collect::<Vec<(_, _)>>();

    // Ethereum block gas limit is 30 million, so it's safe to cast to f64.
    let gas_limit = block.header().gas_limit as f64;

    Ok(reward_percentiles
        .iter()
        .map(|percentile| {
            let mut gas_used = 0;
            let target_gas = ((percentile.as_ref() / 100.0) * gas_limit) as u64;

            for (gas_used_by_tx, effective_reward) in &gas_used_and_effective_reward {
                gas_used += gas_used_by_tx;
                if target_gas <= gas_used {
                    return *effective_reward;
                }
            }

            gas_used_and_effective_reward
                .last()
                .map_or(U256::ZERO, |(_, reward)| *reward)
        })
        .collect())
}

/// Gas used to gas limit ratio
pub(super) fn gas_used_ratio(gas_used: u64, gas_limit: u64) -> f64 {
    // Ported from Hardhat
    // https://github.com/NomicFoundation/hardhat/blob/0c547784952d6409e157b03ae69ba456b03cf6ee/packages/hardhat-core/src/internal/hardhat-network/provider/node.ts#L1359
    const FLOATS_PRECISION: f64 = 100_000.0;
    gas_used as f64 * FLOATS_PRECISION / gas_limit as f64 / FLOATS_PRECISION
}
