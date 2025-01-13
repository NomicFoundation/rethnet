use std::{cmp::Ordering, fmt::Debug};

use edr_eth::{
    block::{calculate_next_base_fee_per_blob_gas, BlockOptions},
    result::{ExecutionResult, InvalidTransaction},
    signature::SignatureError,
    spec::ChainSpec,
    transaction::{ExecutableTransaction as _, Transaction, TransactionValidation},
    U256,
};
use revm::wiring::HaltReasonTrait;
use serde::{Deserialize, Serialize};

use crate::{
    block::BlockBuilderCreationError,
    blockchain::SyncBlockchain,
    config::CfgEnv,
    debug::DebugContext,
    mempool::OrderedTransaction,
    spec::{RuntimeSpec, SyncRuntimeSpec},
    state::{StateDiff, SyncState},
    transaction::TransactionError,
    Block as _, BlockBuilder, BlockBuilderAndError, BlockTransactionError, MemPool,
};

/// The result of mining a block, including the state. This result needs to be
/// inserted into the blockchain to be persistent.
pub struct MineBlockResultAndState<HaltReasonT: HaltReasonTrait, LocalBlockT, StateErrorT> {
    /// Mined block
    pub block: LocalBlockT,
    /// State after mining the block
    pub state: Box<dyn SyncState<StateErrorT>>,
    /// State diff applied by block
    pub state_diff: StateDiff,
    /// Transaction results
    pub transaction_results: Vec<ExecutionResult<HaltReasonT>>,
}

/// Helper type for a chain-specific [`MineBlockResultAndState`].
pub type MineBlockResultAndStateForChainSpec<ChainSpecT, StateErrorT> = MineBlockResultAndState<
    <ChainSpecT as ChainSpec>::HaltReason,
    <ChainSpecT as RuntimeSpec>::LocalBlock,
    StateErrorT,
>;

/// The type of ordering to use when selecting blocks to mine.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum MineOrdering {
    /// Insertion order
    Fifo,
    /// Effective miner fee
    Priority,
}

/// An error that occurred while mining a block.
#[derive(Debug, thiserror::Error)]
pub enum MineBlockError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: RuntimeSpec,
{
    /// An error that occurred while constructing a block builder.
    #[error(transparent)]
    BlockBuilderCreation(
        #[from] BlockBuilderCreationError<BlockchainErrorT, ChainSpecT::Hardfork, StateErrorT>,
    ),
    /// An error that occurred while executing a transaction.
    #[error(transparent)]
    BlockTransaction(#[from] BlockTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>),
    /// An error that occurred while finalizing a block.
    #[error(transparent)]
    BlockFinalize(StateErrorT),
    /// A blockchain error
    #[error(transparent)]
    Blockchain(BlockchainErrorT),
    /// The block is expected to have a prevrandao, as the executor's config is
    /// on a post-merge hardfork.
    #[error("Post-merge transaction is missing prevrandao")]
    MissingPrevrandao,
}

/// Mines a block using as many transactions as can fit in it.
#[allow(clippy::too_many_arguments)]
// `DebugContext` cannot be simplified further
#[allow(clippy::type_complexity)]
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
pub fn mine_block<'blockchain, 'evm, ChainSpecT, DebugDataT, BlockchainErrorT, StateErrorT>(
    blockchain: &'blockchain dyn SyncBlockchain<ChainSpecT, BlockchainErrorT, StateErrorT>,
    state: Box<dyn SyncState<StateErrorT>>,
    mem_pool: &MemPool<ChainSpecT>,
    cfg: &CfgEnv,
    hardfork: ChainSpecT::Hardfork,
    options: BlockOptions,
    min_gas_price: U256,
    mine_ordering: MineOrdering,
    reward: U256,
    debug_context: Option<
        DebugContext<
            'evm,
            ChainSpecT,
            BlockchainErrorT,
            DebugDataT,
            Box<dyn SyncState<StateErrorT>>,
        >,
    >,
) -> Result<
    MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateErrorT>,
    MineBlockError<ChainSpecT, BlockchainErrorT, StateErrorT>,
>
where
    'blockchain: 'evm,
    ChainSpecT: SyncRuntimeSpec<
        SignedTransaction: TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    BlockchainErrorT: Debug + Send,
    StateErrorT: Debug + Send,
{
    let mut block_builder = ChainSpecT::BlockBuilder::new_block_builder(
        blockchain,
        state,
        hardfork,
        cfg.clone(),
        options,
        debug_context,
    )?;

    let mut pending_transactions = {
        type MineOrderComparator<ChainSpecT> = dyn Fn(&OrderedTransaction<ChainSpecT>, &OrderedTransaction<ChainSpecT>) -> Ordering
            + Send;

        let base_fee = block_builder.header().base_fee;
        let comparator: Box<MineOrderComparator<ChainSpecT>> = match mine_ordering {
            MineOrdering::Fifo => Box::new(first_in_first_out_comparator),
            MineOrdering::Priority => {
                Box::new(move |lhs, rhs| priority_comparator(lhs, rhs, base_fee))
            }
        };

        mem_pool.iter(comparator)
    };

    while let Some(transaction) = pending_transactions.next() {
        if *transaction.gas_price() < min_gas_price {
            pending_transactions.remove_caller(transaction.caller());
            continue;
        }

        let caller = *transaction.caller();

        block_builder = block_builder.add_transaction(transaction).map_or_else(
            |BlockBuilderAndError {
                 block_builder,
                 error,
             }| match error {
                BlockTransactionError::ExceedsBlockGasLimit => {
                    pending_transactions.remove_caller(&caller);

                    Ok(block_builder)
                }
                BlockTransactionError::Transaction(TransactionError::InvalidTransaction(error))
                    if error == InvalidTransaction::GasPriceLessThanBasefee.into() =>
                {
                    pending_transactions.remove_caller(&caller);

                    Ok(block_builder)
                }
                error => Err(MineBlockError::BlockTransaction(error)),
            },
            Ok,
        )?;
    }

    let beneficiary = block_builder.header().beneficiary;
    let rewards = vec![(beneficiary, reward)];

    block_builder
        .finalize(rewards)
        .map_err(MineBlockError::BlockFinalize)
}

/// An error that occurred while mining a block with a single transaction.
#[derive(Debug, thiserror::Error)]
pub enum MineTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: RuntimeSpec,
{
    /// An error that occurred while constructing a block builder.
    #[error(transparent)]
    BlockBuilderCreation(
        #[from] BlockBuilderCreationError<BlockchainErrorT, ChainSpecT::Hardfork, StateErrorT>,
    ),
    /// An error that occurred while executing a transaction.
    #[error(transparent)]
    BlockTransaction(#[from] BlockTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>),
    /// A blockchain error
    #[error(transparent)]
    Blockchain(BlockchainErrorT),
    /// The transaction's gas price is lower than the block's minimum gas price.
    #[error("Transaction gasPrice ({actual}) is too low for the next block, which has a baseFeePerGas of {expected}")]
    GasPriceTooLow {
        /// The minimum gas price.
        expected: U256,
        /// The actual gas price.
        actual: U256,
    },
    /// The transaction's max fee per gas is lower than the next block's base
    /// fee.
    #[error("Transaction maxFeePerGas ({actual}) is too low for the next block, which has a baseFeePerGas of {expected}")]
    MaxFeePerGasTooLow {
        /// The minimum max fee per gas.
        expected: U256,
        /// The actual max fee per gas.
        actual: U256,
    },
    /// The transaction's max fee per blob gas is lower than the next block's
    /// base fee.
    #[error("Transaction maxFeePerBlobGas ({actual}) is too low for the next block, which has a baseFeePerBlobGas of {expected}")]
    MaxFeePerBlobGasTooLow {
        /// The minimum max fee per blob gas.
        expected: U256,
        /// The actual max fee per blob gas.
        actual: U256,
    },
    /// The block is expected to have a prevrandao, as the executor's config is
    /// on a post-merge hardfork.
    #[error("Post-merge transaction is missing prevrandao")]
    MissingPrevrandao,
    /// The transaction nonce is too high.
    #[error("Nonce too high. Expected nonce to be {expected} but got {actual}. Note that transactions can't be queued when automining.")]
    NonceTooHigh {
        /// The expected nonce.
        expected: u64,
        /// The actual nonce.
        actual: u64,
    },
    /// The transaction nonce is too high.
    #[error("Nonce too low. Expected nonce to be {expected} but got {actual}. Note that transactions can't be queued when automining.")]
    NonceTooLow {
        /// The expected nonce.
        expected: u64,
        /// The actual nonce.
        actual: u64,
    },
    /// The transaction's priority fee is lower than the minimum gas price.
    #[error("Transaction gas price is {actual}, which is below the minimum of {expected}")]
    PriorityFeeTooLow {
        /// The minimum gas price.
        expected: U256,
        /// The actual max priority fee per gas.
        actual: U256,
    },
    /// Signature error
    #[error(transparent)]
    Signature(#[from] SignatureError),
    /// An error that occurred while querying state.
    #[error(transparent)]
    State(StateErrorT),
}

/// Mines a block with a single transaction.
///
/// If the transaction is invalid, returns an error.
#[allow(clippy::too_many_arguments)]
// `DebugContext` cannot be simplified further
#[allow(clippy::type_complexity)]
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
pub fn mine_block_with_single_transaction<
    'blockchain,
    'evm,
    ChainSpecT,
    DebugDataT,
    BlockchainErrorT,
    StateErrorT,
>(
    blockchain: &'blockchain dyn SyncBlockchain<ChainSpecT, BlockchainErrorT, StateErrorT>,
    state: Box<dyn SyncState<StateErrorT>>,
    transaction: ChainSpecT::SignedTransaction,
    cfg: &CfgEnv,
    hardfork: ChainSpecT::Hardfork,
    options: BlockOptions,
    min_gas_price: U256,
    reward: U256,
    debug_context: Option<
        DebugContext<
            'evm,
            ChainSpecT,
            BlockchainErrorT,
            DebugDataT,
            Box<dyn SyncState<StateErrorT>>,
        >,
    >,
) -> Result<
    MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateErrorT>,
    MineTransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>,
>
where
    'blockchain: 'evm,
    ChainSpecT: SyncRuntimeSpec<
        SignedTransaction: TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    BlockchainErrorT: Debug + Send,
    StateErrorT: Debug + Send,
{
    let max_priority_fee_per_gas = transaction
        .max_priority_fee_per_gas()
        .unwrap_or_else(|| transaction.gas_price());

    if *max_priority_fee_per_gas < min_gas_price {
        return Err(MineTransactionError::PriorityFeeTooLow {
            expected: min_gas_price,
            actual: *max_priority_fee_per_gas,
        });
    }

    if let Some(base_fee_per_gas) = options.base_fee {
        if let Some(max_fee_per_gas) = transaction.max_fee_per_gas() {
            if *max_fee_per_gas < base_fee_per_gas {
                return Err(MineTransactionError::MaxFeePerGasTooLow {
                    expected: base_fee_per_gas,
                    actual: *max_fee_per_gas,
                });
            }
        } else {
            let gas_price = transaction.gas_price();
            if *gas_price < base_fee_per_gas {
                return Err(MineTransactionError::GasPriceTooLow {
                    expected: base_fee_per_gas,
                    actual: *gas_price,
                });
            }
        }
    }

    let parent_block = blockchain
        .last_block()
        .map_err(MineTransactionError::Blockchain)?;

    if let Some(max_fee_per_blob_gas) = transaction.max_fee_per_blob_gas() {
        let base_fee_per_blob_gas = calculate_next_base_fee_per_blob_gas(parent_block.header());
        if *max_fee_per_blob_gas < base_fee_per_blob_gas {
            return Err(MineTransactionError::MaxFeePerBlobGasTooLow {
                expected: base_fee_per_blob_gas,
                actual: *max_fee_per_blob_gas,
            });
        }
    }

    let sender = state
        .basic(*transaction.caller())
        .map_err(MineTransactionError::State)?
        .unwrap_or_default();

    // TODO: This is also checked by `revm`, so it can be simplified
    match transaction.nonce().cmp(&sender.nonce) {
        Ordering::Less => {
            return Err(MineTransactionError::NonceTooLow {
                expected: sender.nonce,
                actual: transaction.nonce(),
            })
        }
        Ordering::Equal => (),
        Ordering::Greater => {
            return Err(MineTransactionError::NonceTooHigh {
                expected: sender.nonce,
                actual: transaction.nonce(),
            })
        }
    }

    let block_builder = ChainSpecT::BlockBuilder::<
        '_,
        BlockchainErrorT,
        DebugDataT,
        StateErrorT,
    >::new_block_builder(
        blockchain,
        state,
        hardfork,
        cfg.clone(),
        options,
        debug_context,
    )?;

    let beneficiary = block_builder.header().beneficiary;
    let rewards = vec![(beneficiary, reward)];

    block_builder
        .add_transaction(transaction)
        .map_err(|BlockBuilderAndError { error, .. }| error)?
        .finalize(rewards)
        .map_err(MineTransactionError::State)
}

fn effective_miner_fee(transaction: &impl Transaction, base_fee: Option<U256>) -> U256 {
    let max_fee_per_gas = transaction.gas_price();
    let max_priority_fee_per_gas = *transaction
        .max_priority_fee_per_gas()
        .unwrap_or(max_fee_per_gas);

    base_fee.map_or(*max_fee_per_gas, |base_fee| {
        max_priority_fee_per_gas.min(*max_fee_per_gas - base_fee)
    })
}

fn first_in_first_out_comparator<ChainSpecT: RuntimeSpec>(
    lhs: &OrderedTransaction<ChainSpecT>,
    rhs: &OrderedTransaction<ChainSpecT>,
) -> Ordering {
    lhs.order_id().cmp(&rhs.order_id())
}

fn priority_comparator<ChainSpecT: RuntimeSpec>(
    lhs: &OrderedTransaction<ChainSpecT>,
    rhs: &OrderedTransaction<ChainSpecT>,
    base_fee: Option<U256>,
) -> Ordering {
    let effective_miner_fee = move |transaction: &ChainSpecT::SignedTransaction| {
        effective_miner_fee(transaction, base_fee)
    };

    // Invert lhs and rhs to get decreasing order by effective miner fee
    let ordering = effective_miner_fee(rhs.pending()).cmp(&effective_miner_fee(lhs.pending()));

    // If two txs have the same effective miner fee we want to sort them
    // in increasing order by orderId
    if ordering == Ordering::Equal {
        lhs.order_id().cmp(&rhs.order_id())
    } else {
        ordering
    }
}

#[cfg(test)]
mod tests {
    use edr_eth::{account::AccountInfo, Address};

    use super::*;
    use crate::test_utils::{
        dummy_eip1559_transaction, dummy_eip155_transaction_with_price, MemPoolTestFixture,
    };

    #[test]
    fn fifo_ordering() -> anyhow::Result<()> {
        let sender1 = Address::random();
        let sender2 = Address::random();
        let sender3 = Address::random();

        let account_with_balance = AccountInfo {
            balance: U256::from(100_000_000u64),
            ..AccountInfo::default()
        };
        let mut fixture = MemPoolTestFixture::with_accounts(&[
            (sender1, account_with_balance.clone()),
            (sender2, account_with_balance.clone()),
            (sender3, account_with_balance),
        ]);

        let base_fee = Some(U256::from(15));

        let transaction1 = dummy_eip155_transaction_with_price(sender1, 0, U256::from(111))?;
        assert_eq!(effective_miner_fee(&transaction1, base_fee), U256::from(96));
        fixture.add_transaction(transaction1.clone())?;

        let transaction2 = dummy_eip1559_transaction(sender2, 0, U256::from(120), U256::from(100))?;
        assert_eq!(
            effective_miner_fee(&transaction2, base_fee),
            U256::from(100)
        );
        fixture.add_transaction(transaction2.clone())?;

        let transaction3 = dummy_eip1559_transaction(sender3, 0, U256::from(140), U256::from(110))?;
        assert_eq!(
            effective_miner_fee(&transaction3, base_fee),
            U256::from(110)
        );
        fixture.add_transaction(transaction3.clone())?;

        let mut ordered_transactions = fixture.mem_pool.iter(first_in_first_out_comparator);

        assert_eq!(ordered_transactions.next(), Some(transaction1));
        assert_eq!(ordered_transactions.next(), Some(transaction2));
        assert_eq!(ordered_transactions.next(), Some(transaction3));

        Ok(())
    }

    #[test]
    fn priority_ordering_gas_price_without_base_fee() -> anyhow::Result<()> {
        let sender1 = Address::random();
        let sender2 = Address::random();
        let sender3 = Address::random();
        let sender4 = Address::random();

        let account_with_balance = AccountInfo {
            balance: U256::from(100_000_000u64),
            ..AccountInfo::default()
        };
        let mut fixture = MemPoolTestFixture::with_accounts(&[
            (sender1, account_with_balance.clone()),
            (sender2, account_with_balance.clone()),
            (sender3, account_with_balance.clone()),
            (sender4, account_with_balance),
        ]);

        let transaction1 = dummy_eip155_transaction_with_price(sender1, 0, U256::from(123))?;
        fixture.add_transaction(transaction1.clone())?;

        let transaction2 = dummy_eip155_transaction_with_price(sender2, 0, U256::from(1_000))?;
        fixture.add_transaction(transaction2.clone())?;

        // This has the same gasPrice than tx2, but arrived later, so it's placed later
        // in the queue
        let transaction3 = dummy_eip155_transaction_with_price(sender3, 0, U256::from(1_000))?;
        fixture.add_transaction(transaction3.clone())?;

        let transaction4 = dummy_eip155_transaction_with_price(sender4, 0, U256::from(2_000))?;
        fixture.add_transaction(transaction4.clone())?;

        let mut ordered_transactions = fixture
            .mem_pool
            .iter(|lhs, rhs| priority_comparator(lhs, rhs, None));

        assert_eq!(ordered_transactions.next(), Some(transaction4));
        assert_eq!(ordered_transactions.next(), Some(transaction2));
        assert_eq!(ordered_transactions.next(), Some(transaction3));
        assert_eq!(ordered_transactions.next(), Some(transaction1));

        Ok(())
    }

    #[test]
    fn priority_ordering_gas_price_with_base_fee() -> anyhow::Result<()> {
        let sender1 = Address::random();
        let sender2 = Address::random();
        let sender3 = Address::random();
        let sender4 = Address::random();
        let sender5 = Address::random();

        let account_with_balance = AccountInfo {
            balance: U256::from(100_000_000u64),
            ..AccountInfo::default()
        };
        let mut fixture = MemPoolTestFixture::with_accounts(&[
            (sender1, account_with_balance.clone()),
            (sender2, account_with_balance.clone()),
            (sender3, account_with_balance.clone()),
            (sender4, account_with_balance.clone()),
            (sender5, account_with_balance),
        ]);

        let base_fee = Some(U256::from(15));

        let transaction1 = dummy_eip155_transaction_with_price(sender1, 0, U256::from(111))?;
        assert_eq!(effective_miner_fee(&transaction1, base_fee), U256::from(96));
        fixture.add_transaction(transaction1.clone())?;

        let transaction2 = dummy_eip1559_transaction(sender2, 0, U256::from(120), U256::from(100))?;
        assert_eq!(
            effective_miner_fee(&transaction2, base_fee),
            U256::from(100)
        );
        fixture.add_transaction(transaction2.clone())?;

        let transaction3 = dummy_eip1559_transaction(sender3, 0, U256::from(140), U256::from(110))?;
        assert_eq!(
            effective_miner_fee(&transaction3, base_fee),
            U256::from(110)
        );
        fixture.add_transaction(transaction3.clone())?;

        let transaction4 = dummy_eip1559_transaction(sender4, 0, U256::from(140), U256::from(130))?;
        assert_eq!(
            effective_miner_fee(&transaction4, base_fee),
            U256::from(125)
        );
        fixture.add_transaction(transaction4.clone())?;

        let transaction5 = dummy_eip155_transaction_with_price(sender5, 0, U256::from(170))?;
        assert_eq!(
            effective_miner_fee(&transaction5, base_fee),
            U256::from(155)
        );
        fixture.add_transaction(transaction5.clone())?;

        let mut ordered_transactions = fixture
            .mem_pool
            .iter(|lhs, rhs| priority_comparator(lhs, rhs, base_fee));

        assert_eq!(ordered_transactions.next(), Some(transaction5));
        assert_eq!(ordered_transactions.next(), Some(transaction4));
        assert_eq!(ordered_transactions.next(), Some(transaction3));
        assert_eq!(ordered_transactions.next(), Some(transaction2));
        assert_eq!(ordered_transactions.next(), Some(transaction1));

        Ok(())
    }

    #[test]
    fn ordering_remove_caller() -> anyhow::Result<()> {
        let sender1 = Address::random();
        let sender2 = Address::random();
        let sender3 = Address::random();
        let sender4 = Address::random();

        let account_with_balance = AccountInfo {
            balance: U256::from(100_000_000u64),
            ..AccountInfo::default()
        };
        let mut fixture = MemPoolTestFixture::with_accounts(&[
            (sender1, account_with_balance.clone()),
            (sender2, account_with_balance.clone()),
            (sender3, account_with_balance.clone()),
            (sender4, account_with_balance),
        ]);

        // Insert 9 transactions sequentially (no for loop)
        let transaction1 = dummy_eip155_transaction_with_price(sender1, 0, U256::from(100))?;
        fixture.add_transaction(transaction1.clone())?;

        let transaction2 = dummy_eip155_transaction_with_price(sender1, 1, U256::from(99))?;
        fixture.add_transaction(transaction2.clone())?;

        let transaction3 = dummy_eip155_transaction_with_price(sender2, 0, U256::from(98))?;
        fixture.add_transaction(transaction3.clone())?;

        let transaction4 = dummy_eip155_transaction_with_price(sender2, 1, U256::from(97))?;
        fixture.add_transaction(transaction4.clone())?;

        let transaction5 = dummy_eip155_transaction_with_price(sender3, 0, U256::from(96))?;
        fixture.add_transaction(transaction5.clone())?;

        let transaction6 = dummy_eip155_transaction_with_price(sender3, 1, U256::from(95))?;
        fixture.add_transaction(transaction6.clone())?;

        let transaction7 = dummy_eip155_transaction_with_price(sender3, 2, U256::from(94))?;
        fixture.add_transaction(transaction7.clone())?;

        let transaction8 = dummy_eip155_transaction_with_price(sender3, 3, U256::from(93))?;
        fixture.add_transaction(transaction8.clone())?;

        let transaction9 = dummy_eip155_transaction_with_price(sender4, 0, U256::from(92))?;
        fixture.add_transaction(transaction9.clone())?;

        let transaction10 = dummy_eip155_transaction_with_price(sender4, 1, U256::from(91))?;
        fixture.add_transaction(transaction10.clone())?;

        let mut ordered_transactions = fixture
            .mem_pool
            .iter(|lhs, rhs| priority_comparator(lhs, rhs, None));

        assert_eq!(ordered_transactions.next(), Some(transaction1));
        assert_eq!(ordered_transactions.next(), Some(transaction2));
        assert_eq!(ordered_transactions.next(), Some(transaction3));

        // Remove all transactions for sender 2
        ordered_transactions.remove_caller(&sender2);

        assert_eq!(ordered_transactions.next(), Some(transaction5));
        assert_eq!(ordered_transactions.next(), Some(transaction6));
        assert_eq!(ordered_transactions.next(), Some(transaction7));

        // Remove all transactions for sender 3
        ordered_transactions.remove_caller(&sender3);

        assert_eq!(ordered_transactions.next(), Some(transaction9));
        assert_eq!(ordered_transactions.next(), Some(transaction10));

        Ok(())
    }
}
