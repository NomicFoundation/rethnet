use std::{collections::BTreeMap, fmt::Debug, num::NonZeroU64, sync::Arc};

use derive_where::derive_where;
use edr_eth::{
    account::{Account, AccountInfo, AccountStatus},
    beacon::{BEACON_ROOTS_ADDRESS, BEACON_ROOTS_BYTECODE},
    block::{largest_safe_block_number, safe_block_depth, LargestSafeBlockNumberArgs},
    l1,
    log::FilterLog,
    spec::HardforkTrait,
    Address, BlockSpec, Bytecode, Bytes, ChainId, HashMap, HashSet, PreEip1898BlockSpec, B256,
    U256,
};
use edr_rpc_eth::{
    client::{EthRpcClient, RpcClientError},
    fork::ForkMetadata,
};
use parking_lot::Mutex;
use tokio::runtime;

use super::{
    compute_state_at_block,
    remote::RemoteBlockchain,
    storage::{
        self, ReservableSparseBlockchainStorage, ReservableSparseBlockchainStorageForChainSpec,
    },
    validate_next_block, BlockHash, Blockchain, BlockchainError, BlockchainErrorForChainSpec,
    BlockchainMut,
};
use crate::{
    block::EthRpcBlock,
    hardfork::Activations,
    spec::{RuntimeSpec, SyncRuntimeSpec},
    state::{ForkState, IrregularState, StateDiff, StateError, StateOverride, SyncState},
    Block, BlockAndTotalDifficulty, BlockAndTotalDifficultyForChainSpec, BlockReceipts,
    RandomHashGenerator, RemoteBlock,
};

/// An error that occurs upon creation of a [`ForkedBlockchain`].
#[derive(Debug, thiserror::Error)]
pub enum CreationError<HardforkT: HardforkTrait> {
    /// JSON-RPC error
    #[error(transparent)]
    RpcClientError(#[from] RpcClientError),
    /// The requested block number does not exist
    #[error("Trying to initialize a provider with block {fork_block_number} but the current block is {latest_block_number}")]
    InvalidBlockNumber {
        /// Requested fork block number
        fork_block_number: u64,
        /// Latest block number
        latest_block_number: u64,
    },
    /// The detected hardfork is not supported
    #[error("Cannot fork {chain_name} from block {fork_block_number}. The hardfork must be at least Spurious Dragon, but {hardfork:?} was detected.")]
    InvalidHardfork {
        /// Requested fork block number
        fork_block_number: u64,
        /// Chain name
        chain_name: String,
        /// Detected hardfork
        hardfork: HardforkT,
    },
}

/// Helper type for a chain-specific [`ForkedBlockchainError`].
pub type ForkedBlockchainErrorForChainSpec<ChainSpecT> = ForkedBlockchainError<
    <ChainSpecT as RuntimeSpec>::RpcBlockConversionError,
    <ChainSpecT as RuntimeSpec>::RpcReceiptConversionError,
>;

/// Error type for [`ForkedBlockchain`].
#[derive(Debug, thiserror::Error)]
pub enum ForkedBlockchainError<BlockConversionErrorT, ReceiptConversionErrorT> {
    /// Remote block creation error
    #[error(transparent)]
    BlockCreation(BlockConversionErrorT),
    /// Remote blocks cannot be deleted
    #[error("Cannot delete remote block.")]
    CannotDeleteRemote,
    /// An error that occurs when trying to insert a block into storage.
    #[error(transparent)]
    Insert(#[from] storage::InsertError),
    /// Rpc client error
    #[error(transparent)]
    RpcClient(#[from] RpcClientError),
    /// Missing transaction receipts for a remote block
    #[error("Missing receipts for block {block_hash}")]
    MissingReceipts {
        /// The block hash
        block_hash: B256,
    },
    /// An error occurred when converting an RPC receipt to an internal type.
    #[error(transparent)]
    ReceiptConversion(ReceiptConversionErrorT),
}

/// A blockchain that forked from a remote blockchain.
#[derive_where(Debug; ChainSpecT::Hardfork)]
pub struct ForkedBlockchain<ChainSpecT>
where
    ChainSpecT: RuntimeSpec,
{
    local_storage: ReservableSparseBlockchainStorageForChainSpec<ChainSpecT>,
    // We can force caching here because we only fork from a safe block number.
    remote: RemoteBlockchain<Arc<RemoteBlock<ChainSpecT>>, ChainSpecT, true>,
    state_root_generator: Arc<Mutex<RandomHashGenerator>>,
    fork_block_number: u64,
    /// The chan id of the forked blockchain is either the local chain id
    /// override or the chain id of the remote blockchain.
    chain_id: u64,
    /// The chain id of the remote blockchain. It might deviate from `chain_id`.
    remote_chain_id: u64,
    network_id: u64,
    hardfork: ChainSpecT::Hardfork,
    hardfork_activations: Option<Activations<ChainSpecT::Hardfork>>,
}

impl<ChainSpecT: RuntimeSpec> ForkedBlockchain<ChainSpecT> {
    /// Constructs a new instance.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        runtime: runtime::Handle,
        chain_id_override: Option<u64>,
        hardfork: ChainSpecT::Hardfork,
        rpc_client: Arc<EthRpcClient<ChainSpecT>>,
        fork_block_number: Option<u64>,
        irregular_state: &mut IrregularState,
        state_root_generator: Arc<Mutex<RandomHashGenerator>>,
        hardfork_activation_overrides: &HashMap<ChainId, Activations<ChainSpecT::Hardfork>>,
    ) -> Result<Self, CreationError<ChainSpecT::Hardfork>> {
        let ForkMetadata {
            chain_id: remote_chain_id,
            network_id,
            latest_block_number,
        } = rpc_client.fork_metadata().await?;

        let recommended_block_number =
            recommended_fork_block_number(RecommendedForkBlockNumberArgs {
                chain_id: remote_chain_id,
                latest_block_number,
            });

        let fork_block_number = if let Some(fork_block_number) = fork_block_number {
            if fork_block_number > latest_block_number {
                return Err(CreationError::InvalidBlockNumber {
                    fork_block_number,
                    latest_block_number,
                });
            }

            if fork_block_number > recommended_block_number {
                let num_confirmations = latest_block_number - fork_block_number + 1;
                let required_confirmations = safe_block_depth(remote_chain_id) + 1;
                let missing_confirmations = required_confirmations - num_confirmations;

                log::warn!("You are forking from block {fork_block_number} which has less than {required_confirmations} confirmations, and will affect Hardhat Network's performance. Please use block number {recommended_block_number} or wait for the block to get {missing_confirmations} more confirmations.");
            }

            fork_block_number
        } else {
            recommended_block_number
        };

        // Dispatch block timestamp request
        let fork_timestamp_future =
            rpc_client.get_block_by_number(PreEip1898BlockSpec::Number(fork_block_number));

        let hardfork_activations = hardfork_activation_overrides
            .get(&remote_chain_id)
            .or_else(|| ChainSpecT::chain_hardfork_activations(remote_chain_id))
            .and_then(|hardfork_activations| {
                // Ignore empty hardfork activations
                if hardfork_activations.is_empty() {
                    None
                } else {
                    Some(hardfork_activations.clone())
                }
            });

        let fork_timestamp = fork_timestamp_future
            .await?
            .expect("Block must exist since block number is less than the latest block number.")
            .timestamp();

        if let Some(remote_hardfork) =
            hardfork_activations
                .as_ref()
                .and_then(|hardfork_activations| {
                    hardfork_activations.hardfork_at_block(fork_block_number, fork_timestamp)
                })
        {
            if remote_hardfork.into() < l1::SpecId::SPURIOUS_DRAGON {
                return Err(CreationError::InvalidHardfork {
                    chain_name: ChainSpecT::chain_name(remote_chain_id)
                        .map_or_else(|| "unknown".to_string(), ToString::to_string),
                    fork_block_number,
                    hardfork: remote_hardfork,
                });
            }

            if remote_hardfork.into() < l1::SpecId::CANCUN && hardfork.into() >= l1::SpecId::CANCUN
            {
                let beacon_roots_contract =
                    Bytecode::new_raw(Bytes::from_static(&BEACON_ROOTS_BYTECODE));

                let state_root = state_root_generator.lock().next_value();

                irregular_state
                    .state_override_at_block_number(fork_block_number)
                    .and_modify(|state_override| {
                        state_override.diff.apply_account_change(
                            BEACON_ROOTS_ADDRESS,
                            AccountInfo {
                                code_hash: beacon_roots_contract.hash_slow(),
                                code: Some(beacon_roots_contract.clone()),
                                ..AccountInfo::default()
                            },
                        );
                    })
                    .or_insert_with(|| {
                        let accounts: HashMap<Address, Account> = [(
                            BEACON_ROOTS_ADDRESS,
                            Account {
                                info: AccountInfo {
                                    code_hash: beacon_roots_contract.hash_slow(),
                                    code: Some(beacon_roots_contract),
                                    ..AccountInfo::default()
                                },
                                status: AccountStatus::Created | AccountStatus::Touched,
                                storage: HashMap::new(),
                            },
                        )]
                        .into_iter()
                        .collect();

                        StateOverride {
                            diff: StateDiff::from(accounts),
                            state_root,
                        }
                    });
            }
        }

        Ok(Self {
            local_storage: ReservableSparseBlockchainStorage::empty(fork_block_number),
            remote: RemoteBlockchain::new(rpc_client, runtime),
            state_root_generator,
            chain_id: chain_id_override.unwrap_or(remote_chain_id),
            remote_chain_id,
            fork_block_number,
            network_id,
            hardfork,
            hardfork_activations,
        })
    }

    /// Returns the chain id of the remote blockchain.
    pub fn remote_chain_id(&self) -> u64 {
        self.remote_chain_id
    }

    fn runtime(&self) -> &runtime::Handle {
        self.remote.runtime()
    }
}

impl<ChainSpecT> Blockchain<ChainSpecT> for ForkedBlockchain<ChainSpecT>
where
    ChainSpecT: SyncRuntimeSpec<
        LocalBlock: BlockReceipts<
            Arc<ChainSpecT::BlockReceipt>,
            Error = BlockchainErrorForChainSpec<ChainSpecT>,
        >,
    >,
{
    type BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>;

    type StateError = StateError;

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    #[allow(clippy::type_complexity)]
    fn block_by_hash(
        &self,
        hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, Self::BlockchainError> {
        if let Some(local_block) = self.local_storage.block_by_hash(hash) {
            Ok(Some(ChainSpecT::cast_local_block(local_block)))
        } else {
            let remote_block = tokio::task::block_in_place(move || {
                self.runtime().block_on(self.remote.block_by_hash(hash))
            })?;

            Ok(remote_block.map(ChainSpecT::cast_remote_block))
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    #[allow(clippy::type_complexity)]
    fn block_by_number(
        &self,
        number: u64,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, Self::BlockchainError> {
        if number <= self.fork_block_number {
            let remote_block = tokio::task::block_in_place(move || {
                self.runtime().block_on(self.remote.block_by_number(number))
            })?;

            Ok(Some(ChainSpecT::cast_remote_block(remote_block)))
        } else {
            let local_block = self.local_storage.block_by_number(number)?;

            Ok(local_block.map(ChainSpecT::cast_local_block))
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    #[allow(clippy::type_complexity)]
    fn block_by_transaction_hash(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, Self::BlockchainError> {
        if let Some(local_block) = self
            .local_storage
            .block_by_transaction_hash(transaction_hash)
        {
            Ok(Some(ChainSpecT::cast_local_block(local_block)))
        } else {
            let remote_block = tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.block_by_transaction_hash(transaction_hash))
            })?;

            Ok(remote_block.map(ChainSpecT::cast_remote_block))
        }
    }

    fn chain_id(&self) -> u64 {
        self.chain_id
    }

    fn chain_id_at_block_number(&self, block_number: u64) -> Result<u64, Self::BlockchainError> {
        if block_number > self.last_block_number() {
            return Err(BlockchainError::UnknownBlockNumber);
        }

        if block_number <= self.fork_block_number {
            Ok(self.remote_chain_id())
        } else {
            Ok(self.chain_id())
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn last_block(&self) -> Result<Arc<ChainSpecT::Block>, Self::BlockchainError> {
        let last_block_number = self.last_block_number();
        if self.fork_block_number < last_block_number {
            let local_block = self
                .local_storage
                .block_by_number(last_block_number)?
                .expect("Block must exist since block number is less than the last block number");

            Ok(ChainSpecT::cast_local_block(local_block))
        } else {
            let remote_block = tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.block_by_number(self.fork_block_number))
            })?;

            Ok(ChainSpecT::cast_remote_block(remote_block))
        }
    }

    fn last_block_number(&self) -> u64 {
        self.local_storage.last_block_number()
    }

    fn logs(
        &self,
        from_block: u64,
        to_block: u64,
        addresses: &HashSet<Address>,
        normalized_topics: &[Option<Vec<B256>>],
    ) -> Result<Vec<FilterLog>, Self::BlockchainError> {
        if from_block <= self.fork_block_number {
            let (to_block, mut local_logs) = if to_block <= self.fork_block_number {
                (to_block, Vec::new())
            } else {
                let local_logs = self.local_storage.logs(
                    self.fork_block_number + 1,
                    to_block,
                    addresses,
                    normalized_topics,
                )?;

                (self.fork_block_number, local_logs)
            };

            let mut remote_logs = tokio::task::block_in_place(move || {
                self.runtime().block_on(self.remote.logs(
                    BlockSpec::Number(from_block),
                    BlockSpec::Number(to_block),
                    addresses,
                    normalized_topics,
                ))
            })?;

            remote_logs.append(&mut local_logs);
            Ok(remote_logs)
        } else {
            self.local_storage
                .logs(from_block, to_block, addresses, normalized_topics)
        }
    }

    fn network_id(&self) -> u64 {
        self.network_id
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn receipt_by_transaction_hash(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::BlockReceipt>>, Self::BlockchainError> {
        if let Some(receipt) = self
            .local_storage
            .receipt_by_transaction_hash(transaction_hash)
        {
            Ok(Some(receipt))
        } else {
            Ok(tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.receipt_by_transaction_hash(transaction_hash))
            })?)
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn spec_at_block_number(
        &self,
        block_number: u64,
    ) -> Result<ChainSpecT::Hardfork, Self::BlockchainError> {
        if block_number > self.last_block_number() {
            return Err(BlockchainError::UnknownBlockNumber);
        }

        if block_number <= self.fork_block_number {
            tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.block_by_number(block_number))
            })
            .map_err(BlockchainError::Forked)
            .and_then(|block| {
                if let Some(hardfork_activations) = &self.hardfork_activations {
                    let header = block.header();
                    hardfork_activations
                        .hardfork_at_block(header.number, header.timestamp)
                        .ok_or(BlockchainError::UnknownBlockSpec {
                            block_number,
                            hardfork_activations: hardfork_activations.clone(),
                        })
                } else {
                    Err(BlockchainError::MissingHardforkActivations {
                        block_number,
                        fork_block_number: self.fork_block_number,
                        chain_id: self.remote_chain_id,
                    })
                }
            })
        } else {
            Ok(self.hardfork)
        }
    }

    fn hardfork(&self) -> ChainSpecT::Hardfork {
        self.hardfork
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn state_at_block_number(
        &self,
        block_number: u64,
        state_overrides: &BTreeMap<u64, StateOverride>,
    ) -> Result<Box<dyn SyncState<Self::StateError>>, Self::BlockchainError> {
        if block_number > self.last_block_number() {
            return Err(BlockchainError::UnknownBlockNumber);
        }

        let state_root = if let Some(state_override) = state_overrides.get(&block_number) {
            state_override.state_root
        } else {
            self.block_by_number(block_number)?
                .expect(
                    "Block must exist since block number is less than equal the last block number.",
                )
                .header()
                .state_root
        };

        let mut state = ForkState::new(
            self.runtime().clone(),
            self.remote.client().clone(),
            self.state_root_generator.clone(),
            block_number,
            state_root,
        );

        let (first_block_number, last_block_number) =
            match block_number.cmp(&self.fork_block_number) {
                // Only override the state at the forked block
                std::cmp::Ordering::Less => (block_number, block_number),
                // Override blocks between the forked block and the requested block
                std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
                    (self.fork_block_number, block_number)
                }
            };

        compute_state_at_block(
            &mut state,
            &self.local_storage,
            first_block_number,
            last_block_number,
            state_overrides,
        );

        // Override the state root in case the local state was modified
        state.set_state_root(state_root);

        Ok(Box::new(state))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn total_difficulty_by_hash(&self, hash: &B256) -> Result<Option<U256>, Self::BlockchainError> {
        if let Some(difficulty) = self.local_storage.total_difficulty_by_hash(hash) {
            Ok(Some(difficulty))
        } else {
            Ok(tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.total_difficulty_by_hash(hash))
            })?)
        }
    }
}

impl<ChainSpecT> BlockchainMut<ChainSpecT> for ForkedBlockchain<ChainSpecT>
where
    ChainSpecT: SyncRuntimeSpec<
        LocalBlock: BlockReceipts<
            Arc<ChainSpecT::BlockReceipt>,
            Error = BlockchainErrorForChainSpec<ChainSpecT>,
        >,
    >,
{
    type Error = BlockchainErrorForChainSpec<ChainSpecT>;

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn insert_block(
        &mut self,
        block: ChainSpecT::LocalBlock,
        state_diff: StateDiff,
    ) -> Result<BlockAndTotalDifficultyForChainSpec<ChainSpecT>, Self::Error> {
        let last_block = self.last_block()?;

        validate_next_block::<ChainSpecT>(self.hardfork, &last_block, &block)?;

        let previous_total_difficulty = self
            .total_difficulty_by_hash(last_block.block_hash())
            .expect("No error can occur as it is stored locally")
            .expect("Must exist as its block is stored");

        let total_difficulty = previous_total_difficulty + block.header().difficulty;

        let block =
            self.local_storage
                .insert_block(Arc::new(block), state_diff, total_difficulty)?;

        Ok(BlockAndTotalDifficulty::new(
            ChainSpecT::cast_local_block(block.clone()),
            Some(total_difficulty),
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn reserve_blocks(&mut self, additional: u64, interval: u64) -> Result<(), Self::Error> {
        let additional = if let Some(additional) = NonZeroU64::new(additional) {
            additional
        } else {
            return Ok(()); // nothing to do
        };

        let last_block = self.last_block()?;
        let previous_total_difficulty = self
            .total_difficulty_by_hash(last_block.block_hash())?
            .expect("Must exist as its block is stored");

        let last_header = last_block.header();
        self.local_storage.reserve_blocks(
            additional,
            interval,
            last_header.base_fee_per_gas,
            last_header.state_root,
            previous_total_difficulty,
            self.hardfork,
        );

        Ok(())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn revert_to_block(&mut self, block_number: u64) -> Result<(), Self::Error> {
        match block_number.cmp(&self.fork_block_number) {
            std::cmp::Ordering::Less => Err(ForkedBlockchainError::CannotDeleteRemote.into()),
            std::cmp::Ordering::Equal => {
                self.local_storage =
                    ReservableSparseBlockchainStorage::empty(self.fork_block_number);

                Ok(())
            }
            std::cmp::Ordering::Greater => {
                if self.local_storage.revert_to_block(block_number) {
                    Ok(())
                } else {
                    Err(BlockchainError::UnknownBlockNumber)
                }
            }
        }
    }
}

impl<ChainSpecT: RuntimeSpec> BlockHash for ForkedBlockchain<ChainSpecT> {
    type Error = BlockchainErrorForChainSpec<ChainSpecT>;

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn block_hash_by_number(&self, block_number: u64) -> Result<B256, Self::Error> {
        if block_number <= self.fork_block_number {
            tokio::task::block_in_place(move || {
                self.runtime()
                    .block_on(self.remote.block_by_number(block_number))
            })
            .map(|block| Ok(*block.block_hash()))?
        } else {
            self.local_storage
                .block_by_number(block_number)?
                .map(|block| *block.block_hash())
                .ok_or(BlockchainError::UnknownBlockNumber)
        }
    }
}

/// Arguments for the `recommended_fork_block_number` function.
/// The purpose of this struct is to prevent mixing up the `u64` arguments.
struct RecommendedForkBlockNumberArgs {
    /// The chain id
    pub chain_id: u64,
    /// The latest known block number
    pub latest_block_number: u64,
}

impl<'a> From<&'a RecommendedForkBlockNumberArgs> for LargestSafeBlockNumberArgs {
    fn from(value: &'a RecommendedForkBlockNumberArgs) -> Self {
        Self {
            chain_id: value.chain_id,
            latest_block_number: value.latest_block_number,
        }
    }
}

/// Determines the recommended block number for forking a specific chain based
/// on the latest block number.
///
/// # Design
///
/// If there is no safe block number, then the latest block number will be used.
/// This decision is based on the assumption that a forked blockchain with a
/// `safe_block_depth` larger than the `latest_block_number` has a high
/// probability of being a devnet.
fn recommended_fork_block_number(args: RecommendedForkBlockNumberArgs) -> u64 {
    largest_safe_block_number(LargestSafeBlockNumberArgs::from(&args))
        .unwrap_or(args.latest_block_number)
}

#[cfg(test)]
mod tests {
    const ROPSTEN_CHAIN_ID: u64 = 3;

    use super::*;

    #[test]
    fn recommended_fork_block_number_with_safe_blocks() {
        const LATEST_BLOCK_NUMBER: u64 = 1_000;

        let safe_block_depth = safe_block_depth(ROPSTEN_CHAIN_ID);
        let args = RecommendedForkBlockNumberArgs {
            chain_id: ROPSTEN_CHAIN_ID,
            latest_block_number: LATEST_BLOCK_NUMBER,
        };
        assert_eq!(
            recommended_fork_block_number(args),
            LATEST_BLOCK_NUMBER - safe_block_depth
        );
    }

    #[test]
    fn recommended_fork_block_number_all_blocks_unsafe() {
        const LATEST_BLOCK_NUMBER: u64 = 50;

        let args = RecommendedForkBlockNumberArgs {
            chain_id: ROPSTEN_CHAIN_ID,
            latest_block_number: LATEST_BLOCK_NUMBER,
        };
        assert_eq!(recommended_fork_block_number(args), LATEST_BLOCK_NUMBER);
    }
}
