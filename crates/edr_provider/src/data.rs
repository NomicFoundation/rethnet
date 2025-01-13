mod account;
mod call;
mod gas;

use std::{
    cmp::{self, Ordering},
    collections::BTreeMap,
    ffi::OsString,
    fmt::Debug,
    num::{NonZeroU64, NonZeroUsize},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use alloy_dyn_abi::eip712::TypedData;
use edr_eth::{
    account::{Account, AccountInfo},
    block::{
        calculate_next_base_fee_per_blob_gas, calculate_next_base_fee_per_gas, miner_reward,
        BlockOptions,
    },
    fee_history::FeeHistoryResult,
    filter::{FilteredEvents, LogOutput, SubscriptionType},
    l1,
    log::FilterLog,
    receipt::{ExecutionReceipt, ReceiptTrait as _},
    result::{ExecutionResult, InvalidTransaction},
    reward_percentile::RewardPercentile,
    signature::{self, RecoveryMessage},
    spec::{ChainSpec, HaltReasonTrait},
    transaction::{
        request::TransactionRequestAndSender,
        signed::{FakeSign as _, Sign as _},
        ExecutableTransaction, IsEip4844, IsSupported as _, Transaction as _, TransactionMut,
        TransactionType, TransactionValidation,
    },
    Address, BlockSpec, BlockTag, Bytecode, Bytes, Eip1898BlockSpec, HashMap, HashSet, B256,
    KECCAK_EMPTY, U256,
};
use edr_evm::{
    block::transaction::{
        BlockDataForTransaction, TransactionAndBlock, TransactionAndBlockForChainSpec,
    },
    blockchain::{
        Blockchain, BlockchainError, BlockchainErrorForChainSpec, ForkedBlockchain,
        ForkedCreationError, GenesisBlockOptions, LocalBlockchain, LocalCreationError,
        SyncBlockchain,
    },
    config::CfgEnv,
    debug_trace_transaction, execution_result_to_debug_result, mempool, mine_block,
    mine_block_with_single_transaction,
    precompile::Precompile,
    register_eip_3155_and_raw_tracers_handles,
    spec::{BlockEnvConstructor as _, RuntimeSpec, SyncRuntimeSpec},
    state::{
        AccountModifierFn, EvmStorageSlot, IrregularState, StateDiff, StateError, StateOverride,
        StateOverrides, SyncState,
    },
    trace::Trace,
    transaction, Block, BlockAndTotalDifficulty, BlockReceipts as _, DebugContext,
    DebugTraceConfig, DebugTraceResultWithTraces, Eip3155AndRawTracers, MemPool,
    MineBlockResultAndState, OrderedTransaction, RandomHashGenerator,
};
use edr_rpc_eth::{
    client::{EthRpcClient, HeaderMap, RpcClientError},
    error::HttpError,
};
use edr_solidity::contract_decoder::{ContractDecoder, ContractDecoderError};
use gas::gas_used_ratio;
use indexmap::IndexMap;
use itertools::izip;
use lru::LruCache;
use revm_precompile::secp256r1;
use rpds::HashTrieMapSync;
use tokio::runtime;

use self::account::{create_accounts, InitialAccounts};
use crate::{
    data::{
        call::{run_call, RunCallArgs},
        gas::{compute_rewards, BinarySearchEstimationArgs, CheckGasLimitArgs},
    },
    debug_mine::{
        DebugMineBlockResult, DebugMineBlockResultAndState, DebugMineBlockResultForChainSpec,
    },
    debugger::{register_debugger_handles, Debugger},
    error::{EstimateGasFailure, TransactionFailure, TransactionFailureWithTraces},
    filter::{bloom_contains_log_filter, filter_logs, Filter, FilterData, LogFilter},
    logger::SyncLogger,
    mock::{Mocker, SyncCallOverride},
    pending::BlockchainWithPending,
    requests::hardhat::rpc_types::{ForkConfig, ForkMetadata},
    snapshot::Snapshot,
    spec::{ProviderSpec, SyncProviderSpec},
    time::{CurrentTime, TimeSinceEpoch},
    MiningConfig, ProviderConfig, ProviderError, SubscriptionEvent, SubscriptionEventData,
    SyncSubscriberCallback,
};

const DEFAULT_INITIAL_BASE_FEE_PER_GAS: u64 = 1_000_000_000;
const EDR_MAX_CACHED_STATES_ENV_VAR: &str = "__EDR_MAX_CACHED_STATES";
const DEFAULT_MAX_CACHED_STATES: usize = 100_000;
const EDR_UNSAFE_SKIP_UNSUPPORTED_TRANSACTION_TYPES: &str =
    "__EDR_UNSAFE_SKIP_UNSUPPORTED_TRANSACTION_TYPES";
const DEFAULT_SKIP_UNSUPPORTED_TRANSACTION_TYPES: bool = false;

/// The result of executing an `eth_call`.
#[derive(Clone, Debug)]
pub struct CallResult<HaltReasonT: HaltReasonTrait> {
    pub console_log_inputs: Vec<Bytes>,
    pub execution_result: ExecutionResult<HaltReasonT>,
    pub trace: Trace<HaltReasonT>,
}

#[derive(Clone)]
pub struct EstimateGasResult<HaltReasonT: HaltReasonTrait> {
    pub estimation: u64,
    pub traces: Vec<Trace<HaltReasonT>>,
}

/// Helper type for a chain-specific [`SendTransactionResult`].
pub type SendTransactionResultForChainSpec<ChainSpecT> = SendTransactionResult<
    Arc<<ChainSpecT as RuntimeSpec>::Block>,
    <ChainSpecT as ChainSpec>::HaltReason,
    <ChainSpecT as ChainSpec>::SignedTransaction,
>;

pub struct SendTransactionResult<BlockT, HaltReasonT: HaltReasonTrait, SignedTransactionT> {
    pub transaction_hash: B256,
    pub mining_results: Vec<DebugMineBlockResult<BlockT, HaltReasonT, SignedTransactionT>>,
}

impl<
        BlockT: Block<SignedTransactionT>,
        HaltReasonT: HaltReasonTrait,
        SignedTransactionT: ExecutableTransaction,
    > SendTransactionResult<BlockT, HaltReasonT, SignedTransactionT>
{
    /// Present if the transaction was auto-mined.
    pub fn transaction_result_and_trace(&self) -> Option<ExecutionResultAndTrace<'_, HaltReasonT>> {
        self.mining_results.iter().find_map(|result| {
            izip!(
                result.block.transactions().iter(),
                result.transaction_results.iter(),
                result.transaction_traces.iter()
            )
            .find_map(|(transaction, result, trace)| {
                if *transaction.transaction_hash() == self.transaction_hash {
                    Some((result, trace))
                } else {
                    None
                }
            })
        })
    }
}

impl<BlockT, HaltReasonT: HaltReasonTrait, SignedTransactionT>
    From<SendTransactionResult<BlockT, HaltReasonT, SignedTransactionT>>
    for (B256, Vec<Trace<HaltReasonT>>)
{
    fn from(value: SendTransactionResult<BlockT, HaltReasonT, SignedTransactionT>) -> Self {
        let SendTransactionResult {
            transaction_hash,
            mining_results,
        } = value;

        let traces = mining_results
            .into_iter()
            .flat_map(|result| result.transaction_traces)
            .collect();

        (transaction_hash, traces)
    }
}

/// The result of executing a transaction.
pub type ExecutionResultAndTrace<'provider, HaltReasonT> = (
    &'provider ExecutionResult<HaltReasonT>,
    &'provider Trace<HaltReasonT>,
);

#[derive(Debug, thiserror::Error)]
pub enum CreationError<ChainSpecT>
where
    ChainSpecT: RuntimeSpec,
{
    /// A blockchain error
    #[error(transparent)]
    Blockchain(BlockchainErrorForChainSpec<ChainSpecT>),
    /// A contract decoder error
    #[error(transparent)]
    ContractDecoder(#[from] ContractDecoderError),
    /// An error that occurred while constructing a forked blockchain.
    #[error(transparent)]
    ForkedBlockchainCreation(#[from] ForkedCreationError<ChainSpecT::Hardfork>),
    #[error("Invalid HTTP header name: {0}")]
    InvalidHttpHeaders(HttpError),
    /// Invalid initial date
    #[error("The initial date configuration value {0:?} is before the UNIX epoch")]
    InvalidInitialDate(SystemTime),
    #[error("Invalid max cached states environment variable value: '{0:?}'. Please provide a non-zero integer!")]
    InvalidMaxCachedStates(OsString),
    /// An error that occurred while constructing a local blockchain.
    #[error(transparent)]
    LocalBlockchainCreation(#[from] LocalCreationError),
    /// An error that occured while querying the remote state.
    #[error(transparent)]
    RpcClient(#[from] RpcClientError),
}

pub struct ProviderData<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch = CurrentTime,
> {
    runtime_handle: runtime::Handle,
    initial_config: ProviderConfig<ChainSpecT::Hardfork>,
    blockchain:
        Box<dyn SyncBlockchain<ChainSpecT, BlockchainErrorForChainSpec<ChainSpecT>, StateError>>,
    pub irregular_state: IrregularState,
    mem_pool: MemPool<ChainSpecT>,
    beneficiary: Address,
    custom_precompiles: HashMap<Address, Precompile>,
    min_gas_price: U256,
    parent_beacon_block_root_generator: RandomHashGenerator,
    prev_randao_generator: RandomHashGenerator,
    block_time_offset_seconds: i64,
    fork_metadata: Option<ForkMetadata>,
    // Must be set if the provider is created with a fork config.
    // Hack to get around the type erasure with the dyn blockchain trait.
    rpc_client: Option<Arc<EthRpcClient<ChainSpecT>>>,
    instance_id: B256,
    is_auto_mining: bool,
    next_block_base_fee_per_gas: Option<U256>,
    next_block_timestamp: Option<u64>,
    next_snapshot_id: u64,
    snapshots: BTreeMap<u64, Snapshot<ChainSpecT>>,
    allow_blocks_with_same_timestamp: bool,
    allow_unlimited_contract_size: bool,
    verbose_tracing: bool,
    // Skip unsupported transaction types in `debugTraceTransaction` instead of throwing an error
    skip_unsupported_transaction_types: bool,
    // IndexMap to preserve account order for logging.
    local_accounts: IndexMap<Address, k256::SecretKey>,
    filters: HashMap<U256, Filter>,
    last_filter_id: U256,
    logger:
        Box<dyn SyncLogger<ChainSpecT, BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>>>,
    impersonated_accounts: HashSet<Address>,
    subscriber_callback: Box<dyn SyncSubscriberCallback<ChainSpecT>>,
    timer: TimerT,
    call_override: Option<Arc<dyn SyncCallOverride>>,
    // We need the Arc to let us avoid returning references to the cache entries which need &mut
    // self to get.
    block_state_cache: LruCache<StateId, Arc<Box<dyn SyncState<StateError>>>>,
    current_state_id: StateId,
    block_number_to_state_id: HashTrieMapSync<u64, StateId>,
    contract_decoder: Arc<ContractDecoder>,
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
{
    pub fn accounts(&self) -> impl Iterator<Item = &Address> {
        self.local_accounts.keys()
    }

    /// Adds a filter for new pending transactions to the provider.
    pub fn add_pending_transaction_filter<const IS_SUBSCRIPTION: bool>(&mut self) -> U256 {
        let filter_id = self.next_filter_id();
        self.filters.insert(
            filter_id,
            Filter::new_pending_transaction_filter(IS_SUBSCRIPTION),
        );
        filter_id
    }

    pub fn allow_unlimited_initcode_size(&self) -> bool {
        self.allow_unlimited_contract_size
    }

    /// Whether the provider is configured to bail on call failures.
    pub fn bail_on_call_failure(&self) -> bool {
        self.initial_config.bail_on_call_failure
    }

    /// Whether the provider is configured to bail on transaction failures.
    pub fn bail_on_transaction_failure(&self) -> bool {
        self.initial_config.bail_on_transaction_failure
    }

    /// Retrieves the gas limit of the next block.
    pub fn block_gas_limit(&self) -> u64 {
        self.mem_pool.block_gas_limit().get()
    }

    pub fn coinbase(&self) -> Address {
        self.beneficiary
    }

    /// Get the locked contract decoder.
    pub fn contract_decoder(&self) -> &ContractDecoder {
        &self.contract_decoder
    }

    /// Returns the default caller.
    pub fn default_caller(&self) -> Address {
        self.local_accounts
            .keys()
            .next()
            .copied()
            .unwrap_or(Address::ZERO)
    }

    /// Returns the metadata of the forked blockchain, if it exists.
    pub fn fork_metadata(&self) -> Option<&ForkMetadata> {
        self.fork_metadata.as_ref()
    }

    pub fn get_filter_changes(&mut self, filter_id: &U256) -> Option<FilteredEvents> {
        self.filters.get_mut(filter_id).map(Filter::take_events)
    }

    pub fn impersonate_account(&mut self, address: Address) {
        self.impersonated_accounts.insert(address);
    }

    pub fn increase_block_time(&mut self, increment: u64) -> i64 {
        self.block_time_offset_seconds += i64::try_from(increment).expect("increment too large");
        self.block_time_offset_seconds
    }

    pub fn instance_id(&self) -> &B256 {
        &self.instance_id
    }

    /// Returns whether the miner is mining automatically.
    pub fn is_auto_mining(&self) -> bool {
        self.is_auto_mining
    }

    pub fn logger_mut(
        &mut self,
    ) -> &mut dyn SyncLogger<ChainSpecT, BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>>
    {
        &mut *self.logger
    }

    /// Returns the instance's [`MiningConfig`].
    pub fn mining_config(&self) -> &MiningConfig {
        &self.initial_config.mining
    }

    /// Returns the instance's network ID.
    pub fn network_id(&self) -> String {
        self.initial_config.network_id.to_string()
    }

    pub fn pending_transactions(&self) -> impl Iterator<Item = &ChainSpecT::SignedTransaction> {
        self.mem_pool.transactions()
    }

    pub fn remove_filter(&mut self, filter_id: &U256) -> bool {
        self.remove_filter_impl::</* IS_SUBSCRIPTION */ false>(filter_id)
    }

    pub fn remove_subscription(&mut self, filter_id: &U256) -> bool {
        self.remove_filter_impl::</* IS_SUBSCRIPTION */ true>(filter_id)
    }

    /// Removes the transaction with the provided hash from the mem pool, if it
    /// exists.
    pub fn remove_pending_transaction(
        &mut self,
        transaction_hash: &B256,
    ) -> Option<OrderedTransaction<ChainSpecT>> {
        self.mem_pool.remove_transaction(transaction_hash)
    }

    /// Retrieves the runtime handle.
    pub fn runtime(&self) -> &runtime::Handle {
        &self.runtime_handle
    }

    /// Sets whether the miner should mine automatically.
    pub fn set_auto_mining(&mut self, enabled: bool) {
        self.is_auto_mining = enabled;
    }

    pub fn set_call_override_callback(&mut self, call_override: Option<Arc<dyn SyncCallOverride>>) {
        self.call_override = call_override;
    }

    /// Sets the coinbase.
    pub fn set_coinbase(&mut self, coinbase: Address) {
        self.beneficiary = coinbase;
    }

    pub fn set_verbose_tracing(&mut self, verbose_tracing: bool) {
        self.verbose_tracing = verbose_tracing;
    }

    pub fn stop_impersonating_account(&mut self, address: Address) -> bool {
        self.impersonated_accounts.remove(&address)
    }

    fn add_state_to_cache(
        &mut self,
        state: Box<dyn SyncState<StateError>>,
        block_number: u64,
    ) -> StateId {
        let state_id = self.current_state_id.increment();
        self.block_state_cache.push(state_id, Arc::new(state));
        self.block_number_to_state_id
            .insert_mut(block_number, state_id);
        state_id
    }

    fn next_filter_id(&mut self) -> U256 {
        self.last_filter_id = self
            .last_filter_id
            .checked_add(U256::from(1))
            .expect("filter id starts at zero, so it'll never overflow for U256");
        self.last_filter_id
    }

    /// Notifies subscribers to `FilterData::NewPendingTransactions` about the
    /// pending transaction with the provided hash.
    fn notify_subscribers_about_pending_transaction(&mut self, transaction_hash: &B256) {
        for (filter_id, filter) in self.filters.iter_mut() {
            if let FilterData::NewPendingTransactions(events) = &mut filter.data {
                if filter.is_subscription {
                    (self.subscriber_callback)(SubscriptionEvent {
                        filter_id: *filter_id,
                        result: SubscriptionEventData::NewPendingTransactions(*transaction_hash),
                    });
                } else {
                    events.push(*transaction_hash);
                }
            }
        }
    }

    /// Notifies subscribers to `FilterData::Logs` and `FilterData::NewHeads`
    /// about the mined block.
    fn notify_subscribers_about_mined_block(
        &mut self,
        block_and_total_difficulty: &BlockAndTotalDifficulty<
            Arc<ChainSpecT::Block>,
            ChainSpecT::SignedTransaction,
        >,
    ) -> Result<(), BlockchainErrorForChainSpec<ChainSpecT>> {
        let block = &block_and_total_difficulty.block;
        for (filter_id, filter) in self.filters.iter_mut() {
            match &mut filter.data {
                FilterData::Logs { criteria, logs } => {
                    let bloom = &block.header().logs_bloom;
                    if bloom_contains_log_filter(bloom, criteria) {
                        let receipts = block.fetch_transaction_receipts()?;
                        let new_logs = receipts.iter().flat_map(ExecutionReceipt::transaction_logs);

                        let mut filtered_logs = filter_logs(new_logs, criteria);
                        if filter.is_subscription {
                            (self.subscriber_callback)(SubscriptionEvent {
                                filter_id: *filter_id,
                                result: SubscriptionEventData::Logs(filtered_logs.clone()),
                            });
                        } else {
                            logs.append(&mut filtered_logs);
                        }
                    }
                }
                FilterData::NewHeads(block_hashes) => {
                    if filter.is_subscription {
                        (self.subscriber_callback)(SubscriptionEvent {
                            filter_id: *filter_id,
                            result: SubscriptionEventData::NewHeads(
                                block_and_total_difficulty.clone(),
                            ),
                        });
                    } else {
                        block_hashes.push(*block.block_hash());
                    }
                }
                FilterData::NewPendingTransactions(_) => (),
            }
        }

        // Remove outdated filters
        self.filters.retain(|_, filter| !filter.has_expired());

        Ok(())
    }

    fn remove_filter_impl<const IS_SUBSCRIPTION: bool>(&mut self, filter_id: &U256) -> bool {
        if let Some(filter) = self.filters.get(filter_id) {
            filter.is_subscription == IS_SUBSCRIPTION && self.filters.remove(filter_id).is_some()
        } else {
            false
        }
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
{
    pub fn get_filter_logs(
        &mut self,
        filter_id: &U256,
    ) -> Result<Option<Vec<LogOutput>>, ProviderError<ChainSpecT>> {
        self.filters
            .get_mut(filter_id)
            .map(|filter| {
                if let Some(events) = filter.take_log_events() {
                    Ok(events)
                } else {
                    Err(ProviderError::InvalidFilterSubscriptionType {
                        filter_id: *filter_id,
                        expected: SubscriptionType::Logs,
                        actual: filter.data.subscription_type(),
                    })
                }
            })
            .transpose()
    }

    pub fn revert_to_snapshot(&mut self, snapshot_id: u64) -> bool {
        // Ensure that, if the snapshot exists, we also remove all subsequent snapshots,
        // as they can only be used once in Ganache.
        let mut removed_snapshots = self.snapshots.split_off(&snapshot_id);

        if let Some(snapshot) = removed_snapshots.remove(&snapshot_id) {
            let Snapshot {
                block_number,
                block_number_to_state_id,
                block_time_offset_seconds,
                coinbase,
                irregular_state,
                mem_pool,
                next_block_base_fee_per_gas,
                next_block_timestamp,
                parent_beacon_block_root_generator,
                prev_randao_generator,
                time,
            } = snapshot;

            self.block_number_to_state_id = block_number_to_state_id;

            // We compute a new offset such that:
            // now + new_offset == snapshot_date + old_offset
            let duration_since_snapshot = Instant::now().duration_since(time);
            self.block_time_offset_seconds = block_time_offset_seconds
                + i64::try_from(duration_since_snapshot.as_secs()).expect("duration too large");

            self.beneficiary = coinbase;
            self.blockchain
                .revert_to_block(block_number)
                .expect("Snapshotted block should exist");

            self.irregular_state = irregular_state;
            self.mem_pool = mem_pool;
            self.next_block_base_fee_per_gas = next_block_base_fee_per_gas;
            self.next_block_timestamp = next_block_timestamp;
            self.parent_beacon_block_root_generator = parent_beacon_block_root_generator;
            self.prev_randao_generator = prev_randao_generator;

            true
        } else {
            false
        }
    }

    pub fn sign(
        &self,
        address: &Address,
        message: Bytes,
    ) -> Result<signature::SignatureWithRecoveryId, ProviderError<ChainSpecT>> {
        match self.local_accounts.get(address) {
            Some(secret_key) => Ok(signature::SignatureWithRecoveryId::new(
                &message[..],
                secret_key,
            )?),
            None => Err(ProviderError::UnknownAddress { address: *address }),
        }
    }

    pub fn sign_typed_data_v4(
        &self,
        address: &Address,
        message: &TypedData,
    ) -> Result<signature::SignatureWithRecoveryId, ProviderError<ChainSpecT>> {
        match self.local_accounts.get(address) {
            Some(secret_key) => {
                let hash = message.eip712_signing_hash()?;
                Ok(signature::SignatureWithRecoveryId::new(
                    RecoveryMessage::Hash(hash),
                    secret_key,
                )?)
            }
            None => Err(ProviderError::UnknownAddress { address: *address }),
        }
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<TimerT>,

    TimerT: Clone + TimeSinceEpoch,
{
    pub fn new(
        runtime_handle: runtime::Handle,
        logger: Box<
            dyn SyncLogger<ChainSpecT, BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>>,
        >,
        subscriber_callback: Box<dyn SyncSubscriberCallback<ChainSpecT>>,
        call_override: Option<Arc<dyn SyncCallOverride>>,
        config: ProviderConfig<ChainSpecT::Hardfork>,
        contract_decoder: Arc<ContractDecoder>,
        timer: TimerT,
    ) -> Result<Self, CreationError<ChainSpecT>> {
        let InitialAccounts {
            local_accounts,
            genesis_state,
        } = create_accounts(&config);

        let BlockchainAndState {
            blockchain,
            fork_metadata,
            rpc_client,
            state,
            irregular_state,
            prev_randao_generator,
            block_time_offset_seconds,
            next_block_base_fee_per_gas,
        } = create_blockchain_and_state(runtime_handle.clone(), &config, &timer, genesis_state)?;

        let max_cached_states = get_max_cached_states_from_env()?;
        let mut block_state_cache = LruCache::new(max_cached_states);
        let mut block_number_to_state_id = HashTrieMapSync::default();

        let current_state_id = StateId::default();
        block_state_cache.push(current_state_id, Arc::new(state));
        block_number_to_state_id.insert_mut(blockchain.last_block_number(), current_state_id);

        let allow_blocks_with_same_timestamp = config.allow_blocks_with_same_timestamp;
        let allow_unlimited_contract_size = config.allow_unlimited_contract_size;
        let beneficiary = config.coinbase;
        let block_gas_limit = config.block_gas_limit;
        let is_auto_mining = config.mining.auto_mine;
        let min_gas_price = config.min_gas_price;

        let skip_unsupported_transaction_types = get_skip_unsupported_transaction_types_from_env();

        let parent_beacon_block_root_generator = if let Some(initial_parent_beacon_block_root) =
            &config.initial_parent_beacon_block_root
        {
            RandomHashGenerator::with_value(*initial_parent_beacon_block_root)
        } else {
            RandomHashGenerator::with_seed("randomParentBeaconBlockRootSeed")
        };

        let custom_precompiles = {
            let mut precompiles = HashMap::new();

            if config.enable_rip_7212 {
                // EIP-7212: secp256r1 P256verify
                precompiles.insert(secp256r1::P256VERIFY.0, secp256r1::P256VERIFY.1);
            }

            precompiles
        };

        Ok(Self {
            runtime_handle,
            initial_config: config,
            blockchain,
            irregular_state,
            mem_pool: MemPool::new(block_gas_limit),
            beneficiary,
            custom_precompiles,
            min_gas_price,
            parent_beacon_block_root_generator,
            prev_randao_generator,
            block_time_offset_seconds,
            fork_metadata,
            rpc_client,
            instance_id: B256::random(),
            is_auto_mining,
            next_block_base_fee_per_gas,
            next_block_timestamp: None,
            // Start with 1 to mimic Ganache
            next_snapshot_id: 1,
            snapshots: BTreeMap::new(),
            allow_blocks_with_same_timestamp,
            allow_unlimited_contract_size,
            verbose_tracing: false,
            skip_unsupported_transaction_types,
            local_accounts,
            filters: HashMap::default(),
            last_filter_id: U256::ZERO,
            logger,
            impersonated_accounts: HashSet::new(),
            subscriber_callback,
            timer,
            call_override,
            block_state_cache,
            current_state_id,
            block_number_to_state_id,
            contract_decoder,
        })
    }

    /// Retrieves the last pending nonce of the account corresponding to the
    /// provided address, if it exists.
    pub fn account_next_nonce(
        &mut self,
        address: &Address,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        let state = self.current_state()?;
        mempool::account_next_nonce(&self.mem_pool, &*state, address).map_err(Into::into)
    }

    /// Adds a filter for new blocks to the provider.
    pub fn add_block_filter<const IS_SUBSCRIPTION: bool>(
        &mut self,
    ) -> Result<U256, ProviderError<ChainSpecT>> {
        let block_hash = *self.last_block()?.block_hash();

        let filter_id = self.next_filter_id();
        self.filters.insert(
            filter_id,
            Filter::new_block_filter(block_hash, IS_SUBSCRIPTION),
        );

        Ok(filter_id)
    }

    /// Adds a filter for new logs to the provider.
    pub fn add_log_filter<const IS_SUBSCRIPTION: bool>(
        &mut self,
        criteria: LogFilter,
    ) -> Result<U256, ProviderError<ChainSpecT>> {
        let logs = self
            .blockchain
            .logs(
                criteria.from_block,
                criteria
                    .to_block
                    .unwrap_or(self.blockchain.last_block_number()),
                &criteria.addresses,
                &criteria.normalized_topics,
            )?
            .iter()
            .map(LogOutput::from)
            .collect();

        let filter_id = self.next_filter_id();
        self.filters.insert(
            filter_id,
            Filter::new_log_filter(criteria, logs, IS_SUBSCRIPTION),
        );
        Ok(filter_id)
    }

    /// Fetch a block by block spec.
    /// Returns `None` if the block spec is `pending`.
    /// Returns `ProviderError::InvalidBlockSpec` error if the block spec is a
    /// number or a hash and the block isn't found.
    /// Returns `ProviderError::InvalidBlockTag` error if the block tag is safe
    /// or finalized and block spec is pre-merge.
    // `SyncBlock` cannot be simplified further
    #[allow(clippy::type_complexity)]
    pub fn block_by_block_spec(
        &self,
        block_spec: &BlockSpec,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, ProviderError<ChainSpecT>> {
        let result = match block_spec {
            BlockSpec::Number(block_number) => Some(
                self.blockchain
                    .block_by_number(*block_number)?
                    .ok_or_else(|| ProviderError::InvalidBlockNumberOrHash {
                        block_spec: block_spec.clone(),
                        latest_block_number: self.blockchain.last_block_number(),
                    })?,
            ),
            BlockSpec::Tag(BlockTag::Earliest) => Some(
                self.blockchain
                    .block_by_number(0)?
                    .expect("genesis block should always exist"),
            ),
            // Matching Hardhat behaviour by returning the last block for finalized and safe.
            // https://github.com/NomicFoundation/hardhat/blob/b84baf2d9f5d3ea897c06e0ecd5e7084780d8b6c/packages/hardhat-core/src/internal/hardhat-network/provider/modules/eth.ts#L1395
            BlockSpec::Tag(tag @ (BlockTag::Finalized | BlockTag::Safe)) => {
                if self.evm_spec_id() >= l1::SpecId::MERGE {
                    Some(self.blockchain.last_block()?)
                } else {
                    return Err(ProviderError::InvalidBlockTag {
                        block_tag: *tag,
                        hardfork: self.hardfork(),
                    });
                }
            }
            BlockSpec::Tag(BlockTag::Latest) => Some(self.blockchain.last_block()?),
            BlockSpec::Tag(BlockTag::Pending) => None,
            BlockSpec::Eip1898(Eip1898BlockSpec::Hash {
                block_hash,
                require_canonical: _,
            }) => Some(self.blockchain.block_by_hash(block_hash)?.ok_or_else(|| {
                ProviderError::InvalidBlockNumberOrHash {
                    block_spec: block_spec.clone(),
                    latest_block_number: self.blockchain.last_block_number(),
                }
            })?),
            BlockSpec::Eip1898(Eip1898BlockSpec::Number { block_number }) => Some(
                self.blockchain
                    .block_by_number(*block_number)?
                    .ok_or_else(|| ProviderError::InvalidBlockNumberOrHash {
                        block_spec: block_spec.clone(),
                        latest_block_number: self.blockchain.last_block_number(),
                    })?,
            ),
        };

        Ok(result)
    }

    /// Retrieves the block that contains a transaction with the provided hash,
    /// if it exists.
    pub fn block_by_transaction_hash(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, ProviderError<ChainSpecT>> {
        self.blockchain
            .block_by_transaction_hash(transaction_hash)
            .map_err(ProviderError::Blockchain)
    }

    // `SyncBlock` cannot be simplified further
    #[allow(clippy::type_complexity)]
    pub fn block_by_hash(
        &self,
        block_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::Block>>, ProviderError<ChainSpecT>> {
        self.blockchain
            .block_by_hash(block_hash)
            .map_err(ProviderError::Blockchain)
    }

    pub fn gas_price(&self) -> Result<U256, ProviderError<ChainSpecT>> {
        const PRE_EIP_1559_GAS_PRICE: u64 = 8_000_000_000;
        const SUGGESTED_PRIORITY_FEE_PER_GAS: u64 = 1_000_000_000;

        if let Some(next_block_gas_fee_per_gas) = self.next_block_base_fee_per_gas()? {
            Ok(next_block_gas_fee_per_gas + U256::from(SUGGESTED_PRIORITY_FEE_PER_GAS))
        } else {
            // We return a hardcoded value for networks without EIP-1559
            Ok(U256::from(PRE_EIP_1559_GAS_PRICE))
        }
    }

    pub fn logs(&self, filter: LogFilter) -> Result<Vec<FilterLog>, ProviderError<ChainSpecT>> {
        self.blockchain
            .logs(
                filter.from_block,
                filter
                    .to_block
                    .unwrap_or(self.blockchain.last_block_number()),
                &filter.addresses,
                &filter.normalized_topics,
            )
            .map_err(ProviderError::Blockchain)
    }

    /// Resets the provider to its initial state, with a modified
    /// [`ForkConfig`].
    pub fn reset(
        &mut self,
        fork_config: Option<ForkConfig>,
    ) -> Result<(), CreationError<ChainSpecT>> {
        let mut config = self.initial_config.clone();
        config.fork = fork_config;

        let mut reset_instance = Self::new(
            self.runtime_handle.clone(),
            self.logger.clone(),
            self.subscriber_callback.clone(),
            self.call_override.clone(),
            config,
            // `hardhat_reset` doesn't discard contract metadata added with
            // `hardhat_addCompilationResult`
            Arc::clone(&self.contract_decoder),
            self.timer.clone(),
        )?;

        std::mem::swap(self, &mut reset_instance);

        Ok(())
    }

    pub fn set_account_storage_slot(
        &mut self,
        address: Address,
        index: U256,
        value: U256,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        // We clone to automatically revert in case of subsequent errors.
        let mut modified_state = (*self.current_state()?).clone();
        let old_value = modified_state.set_account_storage_slot(address, index, value)?;

        let slot = EvmStorageSlot::new_changed(old_value, value);
        let account_info = modified_state.basic(address).and_then(|mut account_info| {
            // Retrieve the code if it's not empty. This is needed for the irregular state.
            if let Some(account_info) = &mut account_info {
                if account_info.code_hash != KECCAK_EMPTY {
                    account_info.code = Some(modified_state.code_by_hash(account_info.code_hash)?);
                }
            }

            Ok(account_info)
        })?;

        let state_root = modified_state.state_root()?;

        let block_number = self.blockchain.last_block_number();
        self.irregular_state
            .state_override_at_block_number(block_number)
            .or_insert_with(|| StateOverride::with_state_root(state_root))
            .diff
            .apply_storage_change(address, index, slot, account_info);

        self.add_state_to_cache(modified_state, block_number);

        Ok(())
    }

    pub fn set_balance(
        &mut self,
        address: Address,
        balance: U256,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let mut modified_state = (*self.current_state()?).clone();
        let account_info = modified_state.modify_account(
            address,
            AccountModifierFn::new(Box::new(move |account_balance, _, _| {
                *account_balance = balance;
            })),
        )?;

        let state_root = modified_state.state_root()?;

        self.mem_pool.update(&modified_state)?;

        let block_number = self.blockchain.last_block_number();
        self.irregular_state
            .state_override_at_block_number(block_number)
            .or_insert_with(|| StateOverride::with_state_root(state_root))
            .diff
            .apply_account_change(address, account_info.clone());

        self.add_state_to_cache(modified_state, block_number);

        Ok(())
    }

    /// Sets the gas limit used for mining new blocks.
    pub fn set_block_gas_limit(
        &mut self,
        gas_limit: NonZeroU64,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let state = self.current_state()?;
        self.mem_pool
            .set_block_gas_limit(&*state, gas_limit)
            .map_err(ProviderError::State)
    }

    pub fn set_code(
        &mut self,
        address: Address,
        code: Bytes,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let code = Bytecode::new_raw(code.clone());
        let irregular_code = code.clone();

        // We clone to automatically revert in case of subsequent errors.
        let mut modified_state = (*self.current_state()?).clone();
        let mut account_info = modified_state.modify_account(
            address,
            AccountModifierFn::new(Box::new(move |_, _, account_code| {
                *account_code = Some(code.clone());
            })),
        )?;

        // The code was stripped from the account, so we need to re-add it for the
        // irregular state.
        account_info.code = Some(irregular_code.clone());

        let state_root = modified_state.state_root()?;

        let block_number = self.blockchain.last_block_number();
        self.irregular_state
            .state_override_at_block_number(block_number)
            .or_insert_with(|| StateOverride::with_state_root(state_root))
            .diff
            .apply_account_change(address, account_info.clone());

        self.add_state_to_cache(modified_state, block_number);

        Ok(())
    }

    pub fn set_min_gas_price(
        &mut self,
        min_gas_price: U256,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        if self.evm_spec_id() >= l1::SpecId::LONDON {
            return Err(ProviderError::SetMinGasPriceUnsupported);
        }

        self.min_gas_price = min_gas_price;

        Ok(())
    }

    /// Sets the next block's base fee per gas.
    pub fn set_next_block_base_fee_per_gas(
        &mut self,
        base_fee_per_gas: U256,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let hardfork = self.hardfork();
        if hardfork.into() < l1::SpecId::LONDON {
            return Err(ProviderError::SetNextBlockBaseFeePerGasUnsupported { hardfork });
        }

        self.next_block_base_fee_per_gas = Some(base_fee_per_gas);

        Ok(())
    }

    /// Set the next block timestamp.
    pub fn set_next_block_timestamp(
        &mut self,
        timestamp: u64,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        let latest_block = self.blockchain.last_block()?;
        let latest_block_header = latest_block.header();

        match timestamp.cmp(&latest_block_header.timestamp) {
            Ordering::Less => Err(ProviderError::TimestampLowerThanPrevious {
                proposed: timestamp,
                previous: latest_block_header.timestamp,
            }),
            Ordering::Equal if !self.allow_blocks_with_same_timestamp => {
                Err(ProviderError::TimestampEqualsPrevious {
                    proposed: timestamp,
                })
            }
            Ordering::Equal | Ordering::Greater => {
                self.next_block_timestamp = Some(timestamp);
                Ok(timestamp)
            }
        }
    }

    /// Sets the next block's prevrandao.
    pub fn set_next_prev_randao(
        &mut self,
        prev_randao: B256,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let hardfork = self.hardfork();
        if hardfork.into() < l1::SpecId::MERGE {
            return Err(ProviderError::SetNextPrevRandaoUnsupported { hardfork });
        }

        self.prev_randao_generator.set_next(prev_randao);

        Ok(())
    }

    pub fn set_nonce(
        &mut self,
        address: Address,
        nonce: u64,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        if mempool::has_transactions(&self.mem_pool) {
            return Err(ProviderError::SetAccountNonceWithPendingTransactions);
        }

        let previous_nonce = self
            .current_state()?
            .basic(address)?
            .map_or(0, |account| account.nonce);

        if nonce < previous_nonce {
            return Err(ProviderError::SetAccountNonceLowerThanCurrent {
                previous: previous_nonce,
                proposed: nonce,
            });
        }

        // We clone to automatically revert in case of subsequent errors.
        let mut modified_state = (*self.current_state()?).clone();
        let account_info = modified_state.modify_account(
            address,
            AccountModifierFn::new(Box::new(move |_, account_nonce, _| *account_nonce = nonce)),
        )?;

        let state_root = modified_state.state_root()?;

        self.mem_pool.update(&modified_state)?;

        let block_number = self.last_block_number();
        self.irregular_state
            .state_override_at_block_number(block_number)
            .or_insert_with(|| StateOverride::with_state_root(state_root))
            .diff
            .apply_account_change(address, account_info.clone());

        self.add_state_to_cache(modified_state, block_number);

        Ok(())
    }

    pub fn sign_transaction_request(
        &self,
        transaction_request: TransactionRequestAndSender<ChainSpecT::TransactionRequest>,
    ) -> Result<ChainSpecT::SignedTransaction, ProviderError<ChainSpecT>> {
        let TransactionRequestAndSender { request, sender } = transaction_request;

        if self.impersonated_accounts.contains(&sender) {
            let signed_transaction = request.fake_sign(sender);
            transaction::validate(signed_transaction, self.evm_spec_id())
                .map_err(ProviderError::TransactionCreationError)
        } else {
            let secret_key = self
                .local_accounts
                .get(&sender)
                .ok_or(ProviderError::UnknownAddress { address: sender })?;

            // SAFETY: We know the secret key belongs to the sender, as we retrieved it from
            // `local_accounts`.
            let signed_transaction =
                unsafe { request.sign_for_sender_unchecked(secret_key, sender) }?;

            transaction::validate(signed_transaction, self.evm_spec_id())
                .map_err(ProviderError::TransactionCreationError)
        }
    }

    pub fn total_difficulty_by_hash(
        &self,
        hash: &B256,
    ) -> Result<Option<U256>, ProviderError<ChainSpecT>> {
        self.blockchain
            .total_difficulty_by_hash(hash)
            .map_err(ProviderError::Blockchain)
    }

    /// Get a transaction by hash from the blockchain or from the mempool if
    /// it's not mined yet.
    pub fn transaction_by_hash(
        &self,
        hash: &B256,
    ) -> Result<Option<TransactionAndBlockForChainSpec<ChainSpecT>>, ProviderError<ChainSpecT>>
    {
        let transaction = if let Some(tx) = self.mem_pool.transaction_by_hash(hash) {
            Some(TransactionAndBlock {
                transaction: tx.pending().clone(),
                block_data: None,
                is_pending: true,
            })
        } else if let Some(block) = self.blockchain.block_by_transaction_hash(hash)? {
            let tx_index_u64 = self
                .blockchain
                .receipt_by_transaction_hash(hash)?
                .expect("If the transaction was inserted in a block, it must have a receipt")
                .transaction_index();
            let tx_index =
                usize::try_from(tx_index_u64).expect("Indices cannot be larger than usize::MAX");

            let transaction = block
                .transactions()
                .get(tx_index)
                .expect("Transaction index must be valid, since it's from the receipt.")
                .clone();

            Some(TransactionAndBlock {
                transaction,
                block_data: Some(BlockDataForTransaction {
                    block,
                    transaction_index: tx_index_u64,
                }),
                is_pending: false,
            })
        } else {
            None
        };

        Ok(transaction)
    }

    pub fn transaction_receipt(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::BlockReceipt>>, ProviderError<ChainSpecT>> {
        self.blockchain
            .receipt_by_transaction_hash(transaction_hash)
            .map_err(ProviderError::Blockchain)
    }

    fn add_pending_transaction(
        &mut self,
        transaction: ChainSpecT::SignedTransaction,
    ) -> Result<B256, ProviderError<ChainSpecT>> {
        let transaction_hash = *transaction.transaction_hash();

        let state = self.current_state()?;
        // Handles validation
        self.mem_pool.add_transaction(&*state, transaction)?;

        self.notify_subscribers_about_pending_transaction(&transaction_hash);

        Ok(transaction_hash)
    }

    /// Retrieves the block number for the provided block spec, if it exists.
    fn block_number_by_block_spec(
        &self,
        block_spec: &BlockSpec,
    ) -> Result<Option<u64>, ProviderError<ChainSpecT>> {
        let block_number = match block_spec {
            BlockSpec::Number(number) => Some(*number),
            BlockSpec::Tag(BlockTag::Earliest) => Some(0),
            BlockSpec::Tag(tag @ (BlockTag::Finalized | BlockTag::Safe)) => {
                if self.evm_spec_id() >= l1::SpecId::MERGE {
                    Some(self.blockchain.last_block_number())
                } else {
                    return Err(ProviderError::InvalidBlockTag {
                        block_tag: *tag,
                        hardfork: self.hardfork(),
                    });
                }
            }
            BlockSpec::Tag(BlockTag::Latest) => Some(self.blockchain.last_block_number()),
            BlockSpec::Tag(BlockTag::Pending) => None,
            BlockSpec::Eip1898(Eip1898BlockSpec::Hash { block_hash, .. }) => {
                self.blockchain.block_by_hash(block_hash)?.map_or_else(
                    || {
                        Err(ProviderError::InvalidBlockNumberOrHash {
                            block_spec: block_spec.clone(),
                            latest_block_number: self.blockchain.last_block_number(),
                        })
                    },
                    |block| Ok(Some(block.header().number)),
                )?
            }
            BlockSpec::Eip1898(Eip1898BlockSpec::Number { block_number }) => Some(*block_number),
        };

        Ok(block_number)
    }

    /// Creates an EVM configuration with the provided hardfork and chain id
    fn create_evm_config(&self, chain_id: u64) -> CfgEnv {
        let mut cfg_env = CfgEnv::default();
        cfg_env.chain_id = chain_id;
        cfg_env.limit_contract_code_size = if self.allow_unlimited_contract_size {
            Some(usize::MAX)
        } else {
            None
        };
        cfg_env.disable_eip3607 = true;

        cfg_env
    }

    /// Creates a configuration, taking into account the hardfork at the
    /// provided `BlockSpec`.
    pub fn create_evm_config_at_block_spec(
        &self,
        block_spec: &BlockSpec,
    ) -> Result<(CfgEnv, ChainSpecT::Hardfork), ProviderError<ChainSpecT>> {
        let block_number = self.block_number_by_block_spec(block_spec)?;

        let spec_id = if let Some(block_number) = block_number {
            self.spec_at_block_number(block_number, block_spec)?
        } else {
            self.blockchain.hardfork()
        };

        let chain_id = if let Some(block_number) = block_number {
            self.chain_id_at_block_number(block_number, block_spec)?
        } else {
            self.blockchain.chain_id()
        };

        let cfg = self.create_evm_config(chain_id);
        Ok((cfg, spec_id))
    }

    fn current_state(
        &mut self,
    ) -> Result<Arc<Box<dyn SyncState<StateError>>>, ProviderError<ChainSpecT>> {
        self.get_or_compute_state(self.last_block_number())
    }

    fn get_or_compute_state(
        &mut self,
        block_number: u64,
    ) -> Result<Arc<Box<dyn SyncState<StateError>>>, ProviderError<ChainSpecT>> {
        if let Some(state_id) = self.block_number_to_state_id.get(&block_number) {
            // We cannot use `LruCache::try_get_or_insert`, because it needs &mut self, but
            // we would need &self in the callback to reference the blockchain.
            if let Some(state) = self.block_state_cache.get(state_id) {
                return Ok(state.clone());
            }
        };

        let state = self
            .blockchain
            .state_at_block_number(block_number, self.irregular_state.state_overrides())?;
        let state_id = self.add_state_to_cache(state, block_number);
        Ok(self
            .block_state_cache
            .get(&state_id)
            // State must exist, since we just inserted it, and we have exclusive access to
            // the cache due to &mut self.
            .expect("State must exist")
            .clone())
    }

    fn mine_and_commit_block_impl(
        &mut self,
        mine_fn: impl FnOnce(
            &mut ProviderData<ChainSpecT, TimerT>,
            &CfgEnv,
            ChainSpecT::Hardfork,
            BlockOptions,
            &mut Debugger<ChainSpecT::HaltReason>,
        ) -> Result<
            MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
            ProviderError<ChainSpecT>,
        >,
        mut options: BlockOptions,
    ) -> Result<DebugMineBlockResultForChainSpec<ChainSpecT>, ProviderError<ChainSpecT>> {
        let (block_timestamp, new_offset) = self.next_block_timestamp(options.timestamp)?;
        options.timestamp = Some(block_timestamp);

        let result = self.mine_block(mine_fn, options)?;

        let block_and_total_difficulty = self
            .blockchain
            .insert_block(result.block, result.state_diff)
            .map_err(ProviderError::Blockchain)?;

        self.mem_pool
            .update(&result.state)
            .map_err(ProviderError::MemPoolUpdate)?;

        if let Some(new_offset) = new_offset {
            self.block_time_offset_seconds = new_offset;
        }

        // Reset the next block base fee per gas upon successful execution
        self.next_block_base_fee_per_gas.take();

        // Reset next block time stamp
        self.next_block_timestamp.take();

        self.parent_beacon_block_root_generator.generate_next();
        self.prev_randao_generator.generate_next();

        self.notify_subscribers_about_mined_block(&block_and_total_difficulty)?;

        self.add_state_to_cache(
            result.state,
            block_and_total_difficulty.block.header().number,
        );

        Ok(DebugMineBlockResult::new(
            block_and_total_difficulty.block,
            result.transaction_results,
            result.transaction_traces,
            result.console_log_inputs,
        ))
    }

    /// Mines a block using the provided options. If an option has not been
    /// specified, it will be set using the provider's configuration values.
    fn mine_block(
        &mut self,
        mine_fn: impl FnOnce(
            &mut ProviderData<ChainSpecT, TimerT>,
            &CfgEnv,
            ChainSpecT::Hardfork,
            BlockOptions,
            &mut Debugger<ChainSpecT::HaltReason>,
        ) -> Result<
            MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
            ProviderError<ChainSpecT>,
        >,
        mut options: BlockOptions,
    ) -> Result<
        DebugMineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
        ProviderError<ChainSpecT>,
    > {
        options.base_fee = options.base_fee.or(self.next_block_base_fee_per_gas);
        options.beneficiary = Some(options.beneficiary.unwrap_or(self.beneficiary));
        options.gas_limit = Some(options.gas_limit.unwrap_or_else(|| self.block_gas_limit()));

        let evm_config = self.create_evm_config(self.blockchain.chain_id());
        let hardfork = self.blockchain.hardfork();

        let evm_spec_id = hardfork.into();
        if options.mix_hash.is_none() && evm_spec_id >= l1::SpecId::MERGE {
            options.mix_hash = Some(self.prev_randao_generator.next_value());
        }

        if evm_spec_id >= l1::SpecId::CANCUN {
            options.parent_beacon_block_root = options
                .parent_beacon_block_root
                .or_else(|| Some(self.parent_beacon_block_root_generator.next_value()));
        }

        let mut debugger = Debugger::with_mocker(
            Mocker::new(self.call_override.clone()),
            self.verbose_tracing,
        );

        let result = mine_fn(self, &evm_config, hardfork, options, &mut debugger)?;

        let Debugger {
            console_logger,
            trace_collector,
            ..
        } = debugger;

        let traces = trace_collector.into_traces();

        Ok(DebugMineBlockResultAndState::new(
            result,
            traces,
            console_logger.into_encoded_messages(),
        ))
    }

    /// Get the timestamp for the next block.
    /// Ported from <https://github.com/NomicFoundation/hardhat/blob/b84baf2d9f5d3ea897c06e0ecd5e7084780d8b6c/packages/hardhat-core/src/internal/hardhat-network/provider/node.ts#L1942>
    fn next_block_timestamp(
        &self,
        timestamp: Option<u64>,
    ) -> Result<(u64, Option<i64>), ProviderError<ChainSpecT>> {
        let latest_block = self.blockchain.last_block()?;
        let latest_block_header = latest_block.header();

        let current_timestamp =
            i64::try_from(self.timer.since_epoch()).expect("timestamp too large");

        let (mut block_timestamp, mut new_offset) = if let Some(timestamp) = timestamp {
            timestamp.checked_sub(latest_block_header.timestamp).ok_or(
                ProviderError::TimestampLowerThanPrevious {
                    proposed: timestamp,
                    previous: latest_block_header.timestamp,
                },
            )?;

            let offset = i64::try_from(timestamp).expect("timestamp too large") - current_timestamp;
            (timestamp, Some(offset))
        } else if let Some(next_block_timestamp) = self.next_block_timestamp {
            let offset = i64::try_from(next_block_timestamp).expect("timestamp too large")
                - current_timestamp;

            (next_block_timestamp, Some(offset))
        } else {
            let next_timestamp = u64::try_from(current_timestamp + self.block_time_offset_seconds)
                .expect("timestamp must be positive");

            (next_timestamp, None)
        };

        let timestamp_needs_increase = block_timestamp == latest_block_header.timestamp
            && !self.allow_blocks_with_same_timestamp;
        if timestamp_needs_increase {
            block_timestamp += 1;
            if new_offset.is_none() {
                new_offset = Some(self.block_time_offset_seconds + 1);
            }
        }

        Ok((block_timestamp, new_offset))
    }

    /// Wrapper over `Blockchain::spec_at_block_number` that handles error
    /// conversion.
    fn spec_at_block_number(
        &self,
        block_number: u64,
        block_spec: &BlockSpec,
    ) -> Result<ChainSpecT::Hardfork, ProviderError<ChainSpecT>> {
        self.blockchain
            .spec_at_block_number(block_number)
            .map_err(|err| match err {
                BlockchainError::UnknownBlockNumber => ProviderError::InvalidBlockNumberOrHash {
                    block_spec: block_spec.clone(),
                    latest_block_number: self.blockchain.last_block_number(),
                },
                _ => ProviderError::Blockchain(err),
            })
    }

    fn validate_auto_mine_transaction(
        &mut self,
        transaction: &ChainSpecT::SignedTransaction,
    ) -> Result<(), ProviderError<ChainSpecT>> {
        let next_nonce = { self.account_next_nonce(transaction.caller())? };

        match transaction.nonce().cmp(&next_nonce) {
            Ordering::Less => {
                return Err(ProviderError::AutoMineNonceTooLow {
                    expected: next_nonce,
                    actual: transaction.nonce(),
                })
            }
            Ordering::Equal => (),
            Ordering::Greater => {
                return Err(ProviderError::AutoMineNonceTooHigh {
                    expected: next_nonce,
                    actual: transaction.nonce(),
                })
            }
        }

        let max_priority_fee_per_gas = transaction
            .max_priority_fee_per_gas()
            .unwrap_or_else(|| transaction.gas_price());

        if *max_priority_fee_per_gas < self.min_gas_price {
            return Err(ProviderError::AutoMinePriorityFeeTooLow {
                expected: self.min_gas_price,
                actual: *max_priority_fee_per_gas,
            });
        }

        if let Some(next_block_base_fee) = self.next_block_base_fee_per_gas()? {
            if let Some(max_fee_per_gas) = transaction.max_fee_per_gas() {
                if *max_fee_per_gas < next_block_base_fee {
                    return Err(ProviderError::AutoMineMaxFeePerGasTooLow {
                        expected: next_block_base_fee,
                        actual: *max_fee_per_gas,
                    });
                }
            } else {
                let gas_price = transaction.gas_price();
                if *gas_price < next_block_base_fee {
                    return Err(ProviderError::AutoMineGasPriceTooLow {
                        expected: next_block_base_fee,
                        actual: *gas_price,
                    });
                }
            }
        }

        Ok(())
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<TimerT>,

    TimerT: Clone + TimeSinceEpoch,
{
    /// Returns the chain ID.
    pub fn chain_id(&self) -> u64 {
        self.blockchain.chain_id()
    }

    pub fn chain_id_at_block_spec(
        &self,
        block_spec: &BlockSpec,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        let block_number = self.block_number_by_block_spec(block_spec)?;

        let chain_id = if let Some(block_number) = block_number {
            self.chain_id_at_block_number(block_number, block_spec)?
        } else {
            self.blockchain.chain_id()
        };

        Ok(chain_id)
    }

    /// Returns the local EVM's [`l1::SpecId`].
    pub fn evm_spec_id(&self) -> l1::SpecId {
        self.hardfork().into()
    }

    /// Returns the local hardfork.
    pub fn hardfork(&self) -> ChainSpecT::Hardfork {
        self.blockchain.hardfork()
    }

    /// Returns the last block in the blockchain.
    pub fn last_block(
        &self,
    ) -> Result<Arc<ChainSpecT::Block>, BlockchainErrorForChainSpec<ChainSpecT>> {
        self.blockchain.last_block()
    }

    /// Returns the number of the last block in the blockchain.
    pub fn last_block_number(&self) -> u64 {
        self.blockchain.last_block_number()
    }

    /// Makes a snapshot of the instance's state and returns the snapshot ID.
    pub fn make_snapshot(&mut self) -> u64 {
        let id = self.next_snapshot_id;
        self.next_snapshot_id += 1;

        let snapshot = Snapshot {
            block_number: self.blockchain.last_block_number(),
            block_number_to_state_id: self.block_number_to_state_id.clone(),
            block_time_offset_seconds: self.block_time_offset_seconds,
            coinbase: self.beneficiary,
            irregular_state: self.irregular_state.clone(),
            mem_pool: self.mem_pool.clone(),
            next_block_base_fee_per_gas: self.next_block_base_fee_per_gas,
            next_block_timestamp: self.next_block_timestamp,
            parent_beacon_block_root_generator: self.parent_beacon_block_root_generator.clone(),
            prev_randao_generator: self.prev_randao_generator.clone(),
            time: Instant::now(),
        };
        self.snapshots.insert(id, snapshot);

        id
    }

    /// Calculates the next block's base fee per gas.
    pub fn next_block_base_fee_per_gas(
        &self,
    ) -> Result<Option<U256>, BlockchainErrorForChainSpec<ChainSpecT>> {
        if self.evm_spec_id() < l1::SpecId::LONDON {
            return Ok(None);
        }

        self.next_block_base_fee_per_gas
            .map_or_else(
                || {
                    let last_block = self.last_block()?;

                    Ok(calculate_next_base_fee_per_gas::<ChainSpecT>(
                        self.blockchain.hardfork(),
                        last_block.header(),
                    ))
                },
                Ok,
            )
            .map(Some)
    }

    /// Calculates the next block's base fee per blob gas.
    pub fn next_block_base_fee_per_blob_gas(
        &self,
    ) -> Result<Option<U256>, BlockchainErrorForChainSpec<ChainSpecT>> {
        if self.evm_spec_id() < l1::SpecId::CANCUN {
            return Ok(None);
        }

        let last_block = self.last_block()?;
        let base_fee = calculate_next_base_fee_per_blob_gas(last_block.header());

        Ok(Some(U256::from(base_fee)))
    }

    /// Calculates the gas price for the next block.
    pub fn next_gas_price(&self) -> Result<U256, BlockchainErrorForChainSpec<ChainSpecT>> {
        if let Some(next_block_base_fee_per_gas) = self.next_block_base_fee_per_gas()? {
            let suggested_priority_fee_per_gas = U256::from(1_000_000_000u64);
            Ok(next_block_base_fee_per_gas + suggested_priority_fee_per_gas)
        } else {
            // We return a hardcoded value for networks without EIP-1559
            Ok(U256::from(8_000_000_000u64))
        }
    }

    /// Wrapper over `Blockchain::chain_id_at_block_number` that handles error
    /// conversion.
    fn chain_id_at_block_number(
        &self,
        block_number: u64,
        block_spec: &BlockSpec,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        self.blockchain
            .chain_id_at_block_number(block_number)
            .map_err(|err| match err {
                BlockchainError::UnknownBlockNumber => ProviderError::InvalidBlockNumberOrHash {
                    block_spec: block_spec.clone(),
                    latest_block_number: self.blockchain.last_block_number(),
                },
                _ => ProviderError::Blockchain(err),
            })
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,

    TimerT: Clone + TimeSinceEpoch,
{
    /// Returns the balance of the account corresponding to the provided address
    /// at the optionally specified [`BlockSpec`]. Otherwise uses the last
    /// block.
    pub fn balance(
        &mut self,
        address: Address,
        block_spec: Option<&BlockSpec>,
    ) -> Result<U256, ProviderError<ChainSpecT>> {
        self.execute_in_block_context::<Result<U256, ProviderError<ChainSpecT>>>(
            block_spec,
            move |_blockchain, _block, state| {
                Ok(state
                    .basic(address)?
                    .map_or(U256::ZERO, |account| account.balance))
            },
        )?
    }

    pub fn debug_trace_call(
        &mut self,
        transaction: ChainSpecT::SignedTransaction,
        block_spec: &BlockSpec,
        trace_config: DebugTraceConfig,
    ) -> Result<DebugTraceResultWithTraces<ChainSpecT::HaltReason>, ProviderError<ChainSpecT>> {
        let (cfg_env, hardfork) = self.create_evm_config_at_block_spec(block_spec)?;

        let mut tracer = Eip3155AndRawTracers::new(trace_config, self.verbose_tracing);
        let precompiles = self.custom_precompiles.clone();

        self.execute_in_block_context(Some(block_spec), |blockchain, block, state| {
            let result = run_call(RunCallArgs {
                blockchain,
                header: block.header(),
                state,
                state_overrides: &StateOverrides::default(),
                cfg_env,
                hardfork,
                transaction,
                precompiles: &precompiles,
                debug_context: Some(DebugContext {
                    data: &mut tracer,
                    register_handles_fn: register_eip_3155_and_raw_tracers_handles,
                }),
            })?;

            Ok(execution_result_to_debug_result(result, tracer))
        })?
    }

    // Matches Hardhat implementation
    pub fn fee_history(
        &mut self,
        block_count: u64,
        newest_block_spec: &BlockSpec,
        percentiles: Option<Vec<RewardPercentile>>,
    ) -> Result<FeeHistoryResult, ProviderError<ChainSpecT>> {
        if self.evm_spec_id() < l1::SpecId::LONDON {
            return Err(ProviderError::UnmetHardfork {
                actual: self.evm_spec_id(),
                minimum: l1::SpecId::LONDON,
            });
        }

        let latest_block_number = self.last_block_number();
        let pending_block_number = latest_block_number + 1;
        let newest_block_number = self
            .block_by_block_spec(newest_block_spec)?
            // None if pending block
            .map_or(pending_block_number, |block| block.header().number);
        let oldest_block_number = if newest_block_number < block_count {
            0
        } else {
            newest_block_number - block_count + 1
        };
        let last_block_number = newest_block_number + 1;

        let pending_block = if last_block_number >= pending_block_number {
            let DebugMineBlockResultAndState { block, .. } = self.mine_pending_block()?;
            Some(block)
        } else {
            None
        };

        let mut result = FeeHistoryResult::new(oldest_block_number);

        let mut reward_and_percentile = percentiles.and_then(|percentiles| {
            if percentiles.is_empty() {
                None
            } else {
                Some((Vec::default(), percentiles))
            }
        });

        let range_includes_remote_blocks = self.fork_metadata.as_ref().map_or(false, |metadata| {
            oldest_block_number <= metadata.fork_block_number
        });

        if range_includes_remote_blocks {
            let last_remote_block = cmp::min(
                self.fork_metadata
                    .as_ref()
                    .expect("we checked that there is a fork")
                    .fork_block_number,
                last_block_number,
            );
            let remote_block_count = last_remote_block - oldest_block_number + 1;

            let rpc_client = self
                .rpc_client
                .as_ref()
                .expect("we checked that there is a fork");
            let FeeHistoryResult {
                oldest_block: _,
                base_fee_per_gas,
                gas_used_ratio,
                reward: remote_reward,
            } = tokio::task::block_in_place(|| {
                self.runtime_handle.block_on(
                    rpc_client.fee_history(
                        remote_block_count,
                        newest_block_spec.clone(),
                        reward_and_percentile
                            .as_ref()
                            .map(|(_, percentiles)| percentiles.clone()),
                    ),
                )
            })?;

            result.base_fee_per_gas = base_fee_per_gas;
            result.gas_used_ratio = gas_used_ratio;
            if let Some((ref mut reward, _)) = reward_and_percentile.as_mut() {
                if let Some(remote_reward) = remote_reward {
                    *reward = remote_reward;
                }
            }
        }

        let first_local_block = if range_includes_remote_blocks {
            cmp::min(
                self.fork_metadata
                    .as_ref()
                    .expect("we checked that there is a fork")
                    .fork_block_number,
                last_block_number,
            ) + 1
        } else {
            oldest_block_number
        };

        for block_number in first_local_block..=last_block_number {
            if block_number < pending_block_number {
                let block = self
                    .blockchain
                    .block_by_number(block_number)?
                    .expect("Block must exist as i is at most the last block number");

                let header = block.header();
                result
                    .base_fee_per_gas
                    .push(header.base_fee_per_gas.unwrap_or(U256::ZERO));

                if block_number < last_block_number {
                    result
                        .gas_used_ratio
                        .push(gas_used_ratio(header.gas_used, header.gas_limit));

                    if let Some((ref mut reward, percentiles)) = reward_and_percentile.as_mut() {
                        reward.push(compute_rewards(block.as_ref(), percentiles)?);
                    }
                }
            } else if block_number == pending_block_number {
                let next_block_base_fee_per_gas = self
                    .next_block_base_fee_per_gas()?
                    .expect("We checked that EIP-1559 is active");
                result.base_fee_per_gas.push(next_block_base_fee_per_gas);

                if block_number < last_block_number {
                    let block = pending_block.as_ref().expect("We mined the pending block");
                    let header = block.header();
                    result
                        .gas_used_ratio
                        .push(gas_used_ratio(header.gas_used, header.gas_limit));

                    if let Some((ref mut reward, percentiles)) = reward_and_percentile.as_mut() {
                        // We don't compute this for the pending block, as there's no
                        // effective miner fee yet.
                        reward.push(percentiles.iter().map(|_| U256::ZERO).collect());
                    }
                }
            } else if block_number == pending_block_number + 1 {
                let block = pending_block.as_ref().expect("We mined the pending block");
                result
                    .base_fee_per_gas
                    .push(calculate_next_base_fee_per_gas::<ChainSpecT>(
                        self.blockchain.hardfork(),
                        block.header(),
                    ));
            }
        }

        if let Some((reward, _)) = reward_and_percentile {
            result.reward = Some(reward);
        }

        Ok(result)
    }

    pub fn get_code(
        &mut self,
        address: Address,
        block_spec: Option<&BlockSpec>,
    ) -> Result<Bytes, ProviderError<ChainSpecT>> {
        self.execute_in_block_context(block_spec, move |_blockchain, _block, state| {
            let code = state
                .basic(address)?
                .map_or(Ok(Bytes::new()), |account_info| {
                    state.code_by_hash(account_info.code_hash).map(|bytecode| {
                        // The `Bytecode` REVM struct pad the bytecode with 33 bytes of 0s for the
                        // `Checked` and `Analysed` variants. `Bytecode::original_bytes` returns
                        // unpadded version.
                        bytecode.original_bytes()
                    })
                })?;

            Ok(code)
        })?
    }

    pub fn get_storage_at(
        &mut self,
        address: Address,
        index: U256,
        block_spec: Option<&BlockSpec>,
    ) -> Result<U256, ProviderError<ChainSpecT>> {
        self.execute_in_block_context::<Result<U256, ProviderError<ChainSpecT>>>(
            block_spec,
            move |_blockchain, _block, state| Ok(state.storage(address, index)?),
        )?
    }

    pub fn get_transaction_count(
        &mut self,
        address: Address,
        block_spec: Option<&BlockSpec>,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        self.execute_in_block_context::<Result<u64, ProviderError<ChainSpecT>>>(
            block_spec,
            move |_blockchain, _block, state| {
                let nonce = state
                    .basic(address)?
                    .map_or(0, |account_info| account_info.nonce);

                Ok(nonce)
            },
        )?
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub fn interval_mine(&mut self) -> Result<bool, ProviderError<ChainSpecT>> {
        let result = self.mine_and_commit_block(BlockOptions::default())?;

        self.logger
            .log_interval_mined(self.hardfork(), &result)
            .map_err(ProviderError::Logger)?;

        Ok(true)
    }

    /// Mines a block with the provided options, using transactions in the
    /// mempool, and commits it to the blockchain.
    pub fn mine_and_commit_block(
        &mut self,
        options: BlockOptions,
    ) -> Result<DebugMineBlockResultForChainSpec<ChainSpecT>, ProviderError<ChainSpecT>> {
        self.mine_and_commit_block_impl(Self::mine_block_with_mem_pool, options)
    }

    /// Mines `number_of_blocks` blocks with the provided `interval` between
    /// them.
    pub fn mine_and_commit_blocks(
        &mut self,
        number_of_blocks: u64,
        interval: u64,
    ) -> Result<Vec<DebugMineBlockResultForChainSpec<ChainSpecT>>, ProviderError<ChainSpecT>> {
        // There should be at least 2 blocks left for the reservation to work,
        // because we always mine a block after it. But here we use a bigger
        // number to err on the side of safety.
        const MINIMUM_RESERVABLE_BLOCKS: u64 = 6;

        if number_of_blocks == 0 {
            return Ok(Vec::new());
        }

        let mine_block_with_interval =
            |data: &mut ProviderData<ChainSpecT, TimerT>,
             mined_blocks: &mut Vec<DebugMineBlockResultForChainSpec<ChainSpecT>>|
             -> Result<(), ProviderError<ChainSpecT>> {
                let previous_timestamp = mined_blocks
                    .last()
                    .expect("at least one block was mined")
                    .block
                    .header()
                    .timestamp;

                let options = BlockOptions {
                    timestamp: Some(previous_timestamp + interval),
                    ..BlockOptions::default()
                };

                let mined_block = data.mine_and_commit_block(options)?;
                mined_blocks.push(mined_block);

                Ok(())
            };

        // Limit the pre-allocated capacity based on the minimum reservable number of
        // blocks to avoid too large allocations.
        let mut mined_blocks = Vec::with_capacity(
            usize::try_from(number_of_blocks.min(2 * MINIMUM_RESERVABLE_BLOCKS))
                .expect("number of blocks exceeds {u64::MAX}"),
        );

        // we always mine the first block, and we don't apply the interval for it
        mined_blocks.push(self.mine_and_commit_block(BlockOptions::default())?);

        while u64::try_from(mined_blocks.len()).expect("usize cannot be larger than u128")
            < number_of_blocks
            && self.mem_pool.has_pending_transactions()
        {
            mine_block_with_interval(self, &mut mined_blocks)?;
        }

        // If there is at least one remaining block, we mine one. This way, we
        // guarantee that there's an empty block immediately before and after the
        // reservation. This makes the logging easier to get right.
        if u64::try_from(mined_blocks.len()).expect("usize cannot be larger than u128")
            < number_of_blocks
        {
            mine_block_with_interval(self, &mut mined_blocks)?;
        }

        let remaining_blocks = number_of_blocks
            - u64::try_from(mined_blocks.len()).expect("usize cannot be larger than u128");

        if remaining_blocks < MINIMUM_RESERVABLE_BLOCKS {
            for _ in 0..remaining_blocks {
                mine_block_with_interval(self, &mut mined_blocks)?;
            }
        } else {
            let current_state = (*self.current_state()?).clone();

            self.blockchain
                .reserve_blocks(remaining_blocks - 1, interval)?;

            // Ensure there is a cache entry for the last reserved block, to avoid
            // recomputation
            self.add_state_to_cache(current_state, self.last_block_number());

            let previous_timestamp = self.blockchain.last_block()?.header().timestamp;
            let options = BlockOptions {
                timestamp: Some(previous_timestamp + interval),
                ..BlockOptions::default()
            };

            let mined_block = self.mine_and_commit_block(options)?;
            mined_blocks.push(mined_block);
        }

        mined_blocks.shrink_to_fit();

        Ok(mined_blocks)
    }

    /// Mines a pending block, without modifying any values.
    pub fn mine_pending_block(
        &mut self,
    ) -> Result<
        DebugMineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
        ProviderError<ChainSpecT>,
    > {
        let (block_timestamp, _new_offset) = self.next_block_timestamp(None)?;

        // Mining a pending block shouldn't affect the mix hash.
        self.mine_block(
            Self::mine_block_with_mem_pool,
            BlockOptions {
                timestamp: Some(block_timestamp),
                ..BlockOptions::default()
            },
        )
    }

    pub fn nonce(
        &mut self,
        address: &Address,
        block_spec: Option<&BlockSpec>,
        state_overrides: &StateOverrides,
    ) -> Result<u64, ProviderError<ChainSpecT>> {
        state_overrides
            .account_override(address)
            .and_then(|account_override| account_override.nonce)
            .map_or_else(
                || {
                    if matches!(block_spec, Some(BlockSpec::Tag(BlockTag::Pending))) {
                        self.account_next_nonce(address)
                    } else {
                        self.execute_in_block_context(
                            block_spec,
                            move |_blockchain, _block, state| {
                                let nonce =
                                    state.basic(*address)?.map_or(0, |account| account.nonce);

                                Ok(nonce)
                            },
                        )?
                    }
                },
                Ok,
            )
    }

    pub fn run_call(
        &mut self,
        transaction: ChainSpecT::SignedTransaction,
        block_spec: &BlockSpec,
        state_overrides: &StateOverrides,
    ) -> Result<CallResult<ChainSpecT::HaltReason>, ProviderError<ChainSpecT>> {
        let (cfg_env, hardfork) = self.create_evm_config_at_block_spec(block_spec)?;

        let mut debugger = Debugger::with_mocker(
            Mocker::new(self.call_override.clone()),
            self.verbose_tracing,
        );

        let precompiles = self.custom_precompiles.clone();
        self.execute_in_block_context(Some(block_spec), |blockchain, block, state| {
            let execution_result = call::run_call(RunCallArgs {
                blockchain,
                header: block.header(),
                state,
                state_overrides,
                cfg_env,
                hardfork,
                transaction,
                precompiles: &precompiles,
                debug_context: Some(DebugContext {
                    data: &mut debugger,
                    register_handles_fn: register_debugger_handles,
                }),
            })?;

            let Debugger {
                console_logger,
                trace_collector,
                ..
            } = debugger;

            let mut traces = trace_collector.into_traces();
            // Should only have a single raw trace
            assert_eq!(traces.len(), 1);

            Ok(CallResult {
                console_log_inputs: console_logger.into_encoded_messages(),
                execution_result,
                trace: traces.pop().expect("Must have a trace"),
            })
        })?
    }

    fn execute_in_block_context<T>(
        &mut self,
        block_spec: Option<&BlockSpec>,
        function: impl FnOnce(
            &dyn SyncBlockchain<ChainSpecT, BlockchainErrorForChainSpec<ChainSpecT>, StateError>,
            &Arc<ChainSpecT::Block>,
            &Box<dyn SyncState<StateError>>,
        ) -> T,
    ) -> Result<T, ProviderError<ChainSpecT>> {
        let block = if let Some(block_spec) = block_spec {
            self.block_by_block_spec(block_spec)?
        } else {
            Some(self.blockchain.last_block()?)
        };

        if let Some(block) = block {
            let block_header = block.header();
            let block_number = block_header.number;

            let contextual_state = self.get_or_compute_state(block_number)?;

            Ok(function(&*self.blockchain, &block, &contextual_state))
        } else {
            // Block spec is pending
            let result = self.mine_pending_block()?;

            let blockchain =
                BlockchainWithPending::new(&*self.blockchain, result.block, result.state_diff);

            let block = blockchain
                .last_block()
                .expect("The pending block is the last block");

            Ok(function(&blockchain, &block, &result.state))
        }
    }

    fn mine_block_with_mem_pool(
        &mut self,
        config: &CfgEnv,
        hardfork: ChainSpecT::Hardfork,
        options: BlockOptions,
        debugger: &mut Debugger<ChainSpecT::HaltReason>,
    ) -> Result<
        MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
        ProviderError<ChainSpecT>,
    > {
        let state_to_be_modified = (*self.current_state()?).clone();
        let result = mine_block(
            self.blockchain.as_ref(),
            state_to_be_modified,
            &self.mem_pool,
            config,
            hardfork,
            options,
            self.min_gas_price,
            self.initial_config.mining.mem_pool.order,
            miner_reward(hardfork.into()).unwrap_or(U256::ZERO),
            Some(DebugContext {
                data: debugger,
                register_handles_fn: register_debugger_handles,
            }),
        )?;

        Ok(result)
    }

    /// Mines a block with the provided transaction.
    fn mine_block_with_single_transaction(
        &mut self,
        config: &CfgEnv,
        hardfork: ChainSpecT::Hardfork,
        options: BlockOptions,
        transaction: ChainSpecT::SignedTransaction,
        debugger: &mut Debugger<ChainSpecT::HaltReason>,
    ) -> Result<
        MineBlockResultAndState<ChainSpecT::HaltReason, ChainSpecT::LocalBlock, StateError>,
        ProviderError<ChainSpecT>,
    > {
        let state_to_be_modified = (*self.current_state()?).clone();
        let result = mine_block_with_single_transaction(
            self.blockchain.as_ref(),
            state_to_be_modified,
            transaction,
            config,
            hardfork,
            options,
            self.min_gas_price,
            miner_reward(hardfork.into()).unwrap_or(U256::ZERO),
            Some(DebugContext {
                data: debugger,
                register_handles_fn: register_debugger_handles,
            }),
        )?;

        Ok(result)
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionType<Type: IsEip4844>
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    TimerT: Clone + TimeSinceEpoch,
{
    pub fn send_transaction(
        &mut self,
        transaction: ChainSpecT::SignedTransaction,
    ) -> Result<SendTransactionResultForChainSpec<ChainSpecT>, ProviderError<ChainSpecT>> {
        if transaction.transaction_type().is_eip4844() {
            if !self.is_auto_mining || mempool::has_transactions(&self.mem_pool) {
                return Err(ProviderError::BlobMemPoolUnsupported);
            }

            let transaction_hash = *transaction.transaction_hash();

            // Despite not adding the transaction to the mempool, we still notify
            // subscribers
            self.notify_subscribers_about_pending_transaction(&transaction_hash);

            let result = self.mine_and_commit_block_impl(
                move |provider, config, hardfork, options, debugger| {
                    provider.mine_block_with_single_transaction(
                        config,
                        hardfork,
                        options,
                        transaction,
                        debugger,
                    )
                },
                BlockOptions::default(),
            )?;

            return Ok(SendTransactionResult {
                transaction_hash,
                mining_results: vec![result],
            });
        }

        let snapshot_id = if self.is_auto_mining {
            self.validate_auto_mine_transaction(&transaction)?;

            Some(self.make_snapshot())
        } else {
            None
        };

        let transaction_hash = self
            .add_pending_transaction(transaction)
            .inspect_err(|_error| {
                if let Some(snapshot_id) = snapshot_id {
                    self.revert_to_snapshot(snapshot_id);
                }
            })?;

        let mut mining_results = Vec::new();
        snapshot_id
            .map(|snapshot_id| -> Result<(), ProviderError<ChainSpecT>> {
                loop {
                    let result = self
                        .mine_and_commit_block(BlockOptions::default())
                        .inspect_err(|_error| {
                            self.revert_to_snapshot(snapshot_id);
                        })?;

                    let mined_transaction = result.has_transaction(&transaction_hash);

                    mining_results.push(result);

                    if mined_transaction {
                        break;
                    }
                }

                while self.mem_pool.has_pending_transactions() {
                    let result = self
                        .mine_and_commit_block(BlockOptions::default())
                        .inspect_err(|_error| {
                            self.revert_to_snapshot(snapshot_id);
                        })?;

                    mining_results.push(result);
                }

                self.snapshots.remove(&snapshot_id);

                Ok(())
            })
            .transpose()?;

        Ok(SendTransactionResult {
            transaction_hash,
            mining_results,
        })
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Clone + Default,
        SignedTransaction: Default
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,

    TimerT: Clone + TimeSinceEpoch,
{
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn debug_trace_transaction(
        &mut self,
        transaction_hash: &B256,
        trace_config: DebugTraceConfig,
    ) -> Result<DebugTraceResultWithTraces<ChainSpecT::HaltReason>, ProviderError<ChainSpecT>> {
        let block = self
            .blockchain
            .block_by_transaction_hash(transaction_hash)?
            .ok_or_else(|| ProviderError::InvalidTransactionHash(*transaction_hash))?;

        let header = block.header();

        let (cfg_env, hardfork) =
            self.create_evm_config_at_block_spec(&BlockSpec::Number(header.number))?;

        let transactions =
            self.filter_unsupported_transaction_types(block.transactions(), transaction_hash)?;

        let prev_block_number = block.header().number - 1;
        let prev_block_spec = Some(BlockSpec::Number(prev_block_number));
        let verbose_tracing = self.verbose_tracing;

        self.execute_in_block_context(
            prev_block_spec.as_ref(),
            |blockchain, _prev_block, state| {
                let block_env = ChainSpecT::BlockEnv::new_block_env(header, hardfork.into());

                debug_trace_transaction(
                    blockchain,
                    state.clone(),
                    cfg_env,
                    hardfork,
                    trace_config,
                    block_env,
                    transactions,
                    transaction_hash,
                    verbose_tracing,
                )
                .map_err(ProviderError::DebugTrace)
            },
        )?
    }

    /// Filters out transactions with unsupported types and returns the
    /// remaining transactions, if skipping is allowed. Otherwise returns
    /// an error.
    fn filter_unsupported_transaction_types(
        &self,
        transactions: &[ChainSpecT::SignedTransaction],
        transaction_hash: &B256,
    ) -> Result<Vec<ChainSpecT::SignedTransaction>, ProviderError<ChainSpecT>> {
        transactions
            .iter()
            .filter_map(|transaction| {
                if transaction.is_supported_transaction() {
                    Some(Ok(transaction.clone()))
                } else if *transaction.transaction_hash() == *transaction_hash {
                    Some(Err(
                        ProviderError::UnsupportedTransactionTypeForDebugTrace {
                            transaction_hash: *transaction_hash,
                            unsupported_transaction_type: transaction.transaction_type().into(),
                        },
                    ))
                } else if self.skip_unsupported_transaction_types {
                    None
                } else {
                    Some(Err(ProviderError::UnsupportedTransactionTypeInDebugTrace {
                        requested_transaction_hash: *transaction_hash,
                        unsupported_transaction_hash: *transaction.transaction_hash(),
                        unsupported_transaction_type: transaction.transaction_type().into(),
                    }))
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<ChainSpecT, TimerT> ProviderData<ChainSpecT, TimerT>
where
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionMut
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,

    TimerT: Clone + TimeSinceEpoch,
{
    /// Estimate the gas cost of a transaction. Matches Hardhat behavior.
    pub fn estimate_gas(
        &mut self,
        transaction: ChainSpecT::SignedTransaction,
        block_spec: &BlockSpec,
    ) -> Result<EstimateGasResult<ChainSpecT::HaltReason>, ProviderError<ChainSpecT>> {
        let (cfg_env, hardfork) = self.create_evm_config_at_block_spec(block_spec)?;
        // Minimum gas cost that is required for transaction to be included in
        // a block
        let minimum_cost = transaction::initial_cost(&transaction, self.evm_spec_id());

        let state_overrides = StateOverrides::default();

        let mut debugger = Debugger::with_mocker(
            Mocker::new(self.call_override.clone()),
            self.verbose_tracing,
        );

        let precompiles = self.custom_precompiles.clone();
        self.execute_in_block_context(Some(block_spec), |blockchain, block, state| {
            let header = block.header();

            // Measure the gas used by the transaction with optional limit from call request
            // defaulting to block limit. Report errors from initial call as if from
            // `eth_call`.
            let result = call::run_call(RunCallArgs {
                blockchain,
                header,
                state,
                state_overrides: &state_overrides,
                cfg_env: cfg_env.clone(),
                hardfork,
                transaction: transaction.clone(),
                precompiles: &precompiles,
                debug_context: Some(DebugContext {
                    data: &mut debugger,
                    register_handles_fn: register_debugger_handles,
                }),
            })?;

            let Debugger {
                console_logger,
                mut trace_collector,
                ..
            } = debugger;

            let mut initial_estimation = match result {
                ExecutionResult::Success { gas_used, .. } => Ok(gas_used),
                ExecutionResult::Revert { output, .. } => Err(TransactionFailure::revert(
                    output,
                    None,
                    trace_collector
                        .traces()
                        .first()
                        .expect("Must have a trace")
                        .clone(),
                )),
                ExecutionResult::Halt { reason, .. } => Err(TransactionFailure::halt(
                    ChainSpecT::cast_halt_reason(reason),
                    None,
                    trace_collector
                        .traces()
                        .first()
                        .expect("Must have a trace")
                        .clone(),
                )),
            }
            .map_err(|failure| EstimateGasFailure {
                console_log_inputs: console_logger.into_encoded_messages(),
                transaction_failure: TransactionFailureWithTraces {
                    traces: vec![failure.solidity_trace.clone()],
                    failure,
                },
            })?;

            // Ensure that the initial estimation is at least the minimum cost + 1.
            if initial_estimation <= minimum_cost {
                initial_estimation = minimum_cost + 1;
            }

            // Test if the transaction would be successful with the initial estimation
            let success = gas::check_gas_limit(CheckGasLimitArgs {
                blockchain,
                header,
                state,
                state_overrides: &state_overrides,
                cfg_env: cfg_env.clone(),
                hardfork,
                transaction: transaction.clone(),
                gas_limit: initial_estimation,
                precompiles: &precompiles,
                trace_collector: &mut trace_collector,
            })?;

            // Return the initial estimation if it was successful
            if success {
                return Ok(EstimateGasResult {
                    estimation: initial_estimation,
                    traces: trace_collector.into_traces(),
                });
            }

            // Correct the initial estimation if the transaction failed with the actually
            // used gas limit. This can happen if the execution logic is based
            // on the available gas.
            let estimation = gas::binary_search_estimation(BinarySearchEstimationArgs {
                blockchain,
                header,
                state,
                state_overrides: &state_overrides,
                cfg_env: cfg_env.clone(),
                hardfork,
                transaction,
                lower_bound: initial_estimation,
                upper_bound: header.gas_limit,
                precompiles: &precompiles,
                trace_collector: &mut trace_collector,
            })?;

            let traces = trace_collector.into_traces();
            Ok(EstimateGasResult { estimation, traces })
        })?
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub(crate) struct StateId(u64);

impl StateId {
    /// Increment the current state id and return the incremented id.
    fn increment(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

fn block_time_offset_seconds<ChainSpecT: RuntimeSpec>(
    config: &ProviderConfig<ChainSpecT::Hardfork>,
    timer: &impl TimeSinceEpoch,
) -> Result<i64, CreationError<ChainSpecT>> {
    config.initial_date.map_or(Ok(0), |initial_date| {
        let initial_timestamp = i64::try_from(
            initial_date
                .duration_since(UNIX_EPOCH)
                .map_err(|_e| CreationError::InvalidInitialDate(initial_date))?
                .as_secs(),
        )
        .expect("initial date must be representable as i64");

        let current_timestamp = i64::try_from(timer.since_epoch())
            .expect("Current timestamp must be representable as i64");

        Ok(initial_timestamp - current_timestamp)
    })
}

struct BlockchainAndState<ChainSpecT: SyncRuntimeSpec> {
    blockchain:
        Box<dyn SyncBlockchain<ChainSpecT, BlockchainErrorForChainSpec<ChainSpecT>, StateError>>,
    fork_metadata: Option<ForkMetadata>,
    rpc_client: Option<Arc<EthRpcClient<ChainSpecT>>>,
    state: Box<dyn SyncState<StateError>>,
    irregular_state: IrregularState,
    prev_randao_generator: RandomHashGenerator,
    block_time_offset_seconds: i64,
    next_block_base_fee_per_gas: Option<U256>,
}

fn create_blockchain_and_state<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    runtime: runtime::Handle,
    config: &ProviderConfig<ChainSpecT::Hardfork>,
    timer: &impl TimeSinceEpoch,
    mut genesis_state: HashMap<Address, Account>,
) -> Result<BlockchainAndState<ChainSpecT>, CreationError<ChainSpecT>> {
    let mut prev_randao_generator = RandomHashGenerator::with_seed(edr_defaults::MIX_HASH_SEED);

    if let Some(fork_config) = &config.fork {
        let state_root_generator = Arc::new(parking_lot::Mutex::new(
            RandomHashGenerator::with_seed(edr_defaults::STATE_ROOT_HASH_SEED),
        ));

        let http_headers = fork_config
            .http_headers
            .as_ref()
            .map(|headers| HeaderMap::try_from(headers).map_err(CreationError::InvalidHttpHeaders))
            .transpose()?;

        let rpc_client = Arc::new(EthRpcClient::<ChainSpecT>::new(
            &fork_config.json_rpc_url,
            config.cache_dir.clone(),
            http_headers.clone(),
        )?);

        let (blockchain, mut irregular_state) = tokio::task::block_in_place(
            || -> Result<_, ForkedCreationError<ChainSpecT::Hardfork>> {
                let mut irregular_state = IrregularState::default();
                let blockchain = runtime.block_on(ForkedBlockchain::<ChainSpecT>::new(
                    runtime.clone(),
                    Some(config.chain_id),
                    config.hardfork,
                    rpc_client.clone(),
                    fork_config.block_number,
                    &mut irregular_state,
                    state_root_generator.clone(),
                    &config.chains,
                ))?;

                Ok((blockchain, irregular_state))
            },
        )?;

        let fork_block_number = blockchain.last_block_number();

        if !genesis_state.is_empty() {
            let genesis_addresses = genesis_state.keys().cloned().collect::<Vec<_>>();
            let genesis_account_infos = tokio::task::block_in_place(|| {
                runtime.block_on(rpc_client.get_account_infos(
                    &genesis_addresses,
                    Some(BlockSpec::Number(fork_block_number)),
                ))
            })?;

            // Make sure that the nonce and the code of genesis accounts matches the fork
            // state as we only want to overwrite the balance.
            for (address, account_info) in genesis_addresses.into_iter().zip(genesis_account_infos)
            {
                genesis_state.entry(address).and_modify(|account| {
                    let AccountInfo {
                        balance: _,
                        nonce,
                        code,
                        code_hash,
                    } = &mut account.info;

                    *nonce = account_info.nonce;
                    *code = account_info.code;
                    *code_hash = account_info.code_hash;
                });
            }

            irregular_state
                .state_override_at_block_number(fork_block_number)
                .and_modify(|state_override| {
                    // No need to update the state_root, as it could only have been created by the
                    // `ForkedBlockchain` constructor.
                    state_override.diff.apply_diff(genesis_state.clone());
                })
                .or_insert_with(|| {
                    let state_root = state_root_generator.lock().next_value();

                    StateOverride {
                        diff: StateDiff::from(genesis_state),
                        state_root,
                    }
                });
        }

        let state = blockchain
            .state_at_block_number(fork_block_number, irregular_state.state_overrides())
            .expect("Fork state must exist");

        let block_time_offset_seconds = {
            let fork_block_timestamp = UNIX_EPOCH
                + Duration::from_secs(
                    blockchain
                        .last_block()
                        .map_err(CreationError::Blockchain)?
                        .header()
                        .timestamp,
                );

            let elapsed = match timer.since(fork_block_timestamp) {
                Ok(elapsed) => -i128::from(elapsed),
                Err(forward_drift) => i128::from(forward_drift.duration().as_secs()),
            };

            elapsed
                .try_into()
                .expect("Elapsed time since fork block must be representable as i64")
        };

        let next_block_base_fee_per_gas = if config.hardfork.into() >= l1::SpecId::LONDON {
            if let Some(base_fee) = config.initial_base_fee_per_gas {
                Some(base_fee)
            } else {
                let previous_base_fee = blockchain
                    .last_block()
                    .map_err(CreationError::Blockchain)?
                    .header()
                    .base_fee_per_gas;

                if previous_base_fee.is_none() {
                    Some(U256::from(DEFAULT_INITIAL_BASE_FEE_PER_GAS))
                } else {
                    None
                }
            }
        } else {
            None
        };

        Ok(BlockchainAndState {
            fork_metadata: Some(ForkMetadata {
                chain_id: blockchain.remote_chain_id(),
                fork_block_number,
                fork_block_hash: *blockchain
                    .block_by_number(fork_block_number)
                    .map_err(CreationError::Blockchain)?
                    .expect("Fork block must exist")
                    .block_hash(),
            }),
            rpc_client: Some(rpc_client),
            blockchain: Box::new(blockchain),
            state: Box::new(state),
            irregular_state,
            prev_randao_generator,
            block_time_offset_seconds,
            next_block_base_fee_per_gas,
        })
    } else {
        let mix_hash = if config.hardfork.into() >= l1::SpecId::MERGE {
            Some(prev_randao_generator.generate_next())
        } else {
            None
        };

        let blockchain = LocalBlockchain::new(
            StateDiff::from(genesis_state),
            config.chain_id,
            config.hardfork,
            GenesisBlockOptions {
                gas_limit: Some(config.block_gas_limit.get()),
                timestamp: config.initial_date.map(|d| {
                    d.duration_since(UNIX_EPOCH)
                        .expect("initial date must be after UNIX epoch")
                        .as_secs()
                }),
                mix_hash,
                base_fee: config.initial_base_fee_per_gas,
                blob_gas: config.initial_blob_gas.clone(),
            },
        )?;

        let irregular_state = IrregularState::default();
        let state = blockchain
            .state_at_block_number(0, irregular_state.state_overrides())
            .expect("Genesis state must exist");

        let block_time_offset_seconds = block_time_offset_seconds(config, timer)?;

        Ok(BlockchainAndState {
            fork_metadata: None,
            rpc_client: None,
            blockchain: Box::new(blockchain),
            state,
            irregular_state,
            block_time_offset_seconds,
            prev_randao_generator,
            // For local blockchain the initial base fee per gas config option is incorporated as
            // part of the genesis block.
            next_block_base_fee_per_gas: None,
        })
    }
}

fn get_skip_unsupported_transaction_types_from_env() -> bool {
    std::env::var(EDR_UNSAFE_SKIP_UNSUPPORTED_TRANSACTION_TYPES)
        .map_or(DEFAULT_SKIP_UNSUPPORTED_TRANSACTION_TYPES, |s| s == "true")
}

fn get_max_cached_states_from_env<ChainSpecT: RuntimeSpec>(
) -> Result<NonZeroUsize, CreationError<ChainSpecT>> {
    std::env::var(EDR_MAX_CACHED_STATES_ENV_VAR).map_or_else(
        |err| match err {
            std::env::VarError::NotPresent => {
                Ok(NonZeroUsize::new(DEFAULT_MAX_CACHED_STATES).expect("constant is non-zero"))
            }
            std::env::VarError::NotUnicode(s) => Err(CreationError::InvalidMaxCachedStates(s)),
        },
        |s| {
            s.parse()
                .map_err(|_err| CreationError::InvalidMaxCachedStates(s.into()))
        },
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use edr_eth::{hex, l1::L1ChainSpec, transaction::ExecutableTransaction as _};
    use edr_evm::MineOrdering;
    use serde_json::json;

    use super::*;
    use crate::{
        console_log::tests::{deploy_console_log_contract, ConsoleLogTransaction},
        test_utils::{create_test_config, one_ether, ProviderTestFixture},
        MemPoolConfig, MiningConfig, ProviderConfig,
    };

    #[test]
    fn test_local_account_balance() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let account = *fixture
            .provider_data
            .local_accounts
            .keys()
            .next()
            .expect("there are local accounts");

        let last_block_number = fixture.provider_data.last_block_number();
        let block_spec = BlockSpec::Number(last_block_number);

        let balance = fixture.provider_data.balance(account, Some(&block_spec))?;

        assert_eq!(balance, one_ether());

        Ok(())
    }

    #[cfg(feature = "test-remote")]
    #[test]
    fn test_local_account_balance_forked() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_forked(None)?;

        let account = *fixture
            .provider_data
            .local_accounts
            .keys()
            .next()
            .expect("there are local accounts");

        let last_block_number = fixture.provider_data.last_block_number();
        let block_spec = BlockSpec::Number(last_block_number);

        let balance = fixture.provider_data.balance(account, Some(&block_spec))?;

        assert_eq!(balance, one_ether());

        Ok(())
    }

    #[test]
    fn test_sign_transaction_request() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction = fixture.signed_dummy_transaction(0, None)?;
        let recovered_address = transaction.caller();

        assert!(fixture
            .provider_data
            .local_accounts
            .contains_key(recovered_address));

        Ok(())
    }

    #[test]
    fn test_sign_transaction_request_impersonated_account() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction = fixture.impersonated_dummy_transaction()?;

        assert_eq!(transaction.caller(), &fixture.impersonated_account);

        Ok(())
    }

    fn test_add_pending_transaction(
        fixture: &mut ProviderTestFixture<L1ChainSpec>,
        transaction: transaction::Signed,
    ) -> anyhow::Result<()> {
        let filter_id = fixture
            .provider_data
            .add_pending_transaction_filter::<false>();

        let transaction_hash = fixture.provider_data.add_pending_transaction(transaction)?;

        assert!(fixture
            .provider_data
            .mem_pool
            .transaction_by_hash(&transaction_hash)
            .is_some());

        match fixture
            .provider_data
            .get_filter_changes(&filter_id)
            .unwrap()
        {
            FilteredEvents::NewPendingTransactions(hashes) => {
                assert!(hashes.contains(&transaction_hash));
            }
            _ => panic!("expected pending transaction"),
        };

        assert!(fixture.provider_data.mem_pool.has_pending_transactions());

        Ok(())
    }

    #[test]
    fn add_pending_transaction() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        let transaction = fixture.signed_dummy_transaction(0, None)?;

        test_add_pending_transaction(&mut fixture, transaction)
    }

    #[test]
    fn add_pending_transaction_from_impersonated_account() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        let transaction = fixture.impersonated_dummy_transaction()?;

        test_add_pending_transaction(&mut fixture, transaction)
    }

    #[test]
    fn block_by_block_spec_earliest() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let block_spec = BlockSpec::Tag(BlockTag::Earliest);

        let block = fixture
            .provider_data
            .block_by_block_spec(&block_spec)?
            .context("block should exist")?;

        assert_eq!(block.header().number, 0);

        Ok(())
    }

    #[test]
    fn block_by_block_spec_finalized_safe_latest() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        // Mine a block to make sure we're not getting the genesis block
        fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;
        let last_block_number = fixture.provider_data.last_block_number();
        // Sanity check
        assert!(last_block_number > 0);

        let block_tags = vec![BlockTag::Finalized, BlockTag::Safe, BlockTag::Latest];
        for tag in block_tags {
            let block_spec = BlockSpec::Tag(tag);

            let block = fixture
                .provider_data
                .block_by_block_spec(&block_spec)?
                .context("block should exist")?;

            assert_eq!(block.header().number, last_block_number);
        }

        Ok(())
    }

    #[test]
    fn block_by_block_spec_pending() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let block_spec = BlockSpec::Tag(BlockTag::Pending);

        let block = fixture.provider_data.block_by_block_spec(&block_spec)?;

        assert!(block.is_none());

        Ok(())
    }

    // Make sure executing a transaction in a pending block context doesn't panic.
    #[test]
    fn execute_in_block_context_pending() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let block_spec = Some(BlockSpec::Tag(BlockTag::Pending));

        let mut value = 0;
        let _ =
            fixture
                .provider_data
                .execute_in_block_context(block_spec.as_ref(), |_, _, _| {
                    value += 1;
                    Ok::<(), ProviderError<L1ChainSpec>>(())
                })?;

        assert_eq!(value, 1);

        Ok(())
    }

    #[test]
    fn chain_id() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let chain_id = fixture.provider_data.chain_id();
        assert_eq!(chain_id, fixture.config.chain_id);

        Ok(())
    }

    #[cfg(feature = "test-remote")]
    #[test]
    fn chain_id_fork_mode() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_forked(None)?;

        let chain_id = fixture.provider_data.chain_id();
        assert_eq!(chain_id, fixture.config.chain_id);

        let chain_id_at_block = fixture
            .provider_data
            .chain_id_at_block_spec(&BlockSpec::Number(1))?;
        assert_eq!(chain_id_at_block, 1);

        Ok(())
    }

    #[cfg(feature = "test-remote")]
    #[test]
    fn fork_metadata_fork_mode() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_forked(None)?;

        let fork_metadata = fixture
            .provider_data
            .fork_metadata()
            .expect("fork metadata should exist");
        assert_eq!(fork_metadata.chain_id, 1);

        Ok(())
    }

    #[test]
    fn console_log_mine_block() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        let ConsoleLogTransaction {
            transaction,
            expected_call_data,
        } = deploy_console_log_contract(&mut fixture.provider_data)?;

        let signed_transaction = fixture
            .provider_data
            .sign_transaction_request(transaction)?;

        fixture.provider_data.set_auto_mining(false);
        fixture.provider_data.send_transaction(signed_transaction)?;
        let (block_timestamp, _) = fixture.provider_data.next_block_timestamp(None)?;
        let prevrandao = fixture.provider_data.prev_randao_generator.next_value();
        let result = fixture.provider_data.mine_block(
            ProviderData::mine_block_with_mem_pool,
            BlockOptions {
                timestamp: Some(block_timestamp),
                mix_hash: Some(prevrandao),
                ..BlockOptions::default()
            },
        )?;

        let console_log_inputs = result.console_log_inputs;
        assert_eq!(console_log_inputs.len(), 1);
        assert_eq!(console_log_inputs[0], expected_call_data);

        Ok(())
    }

    #[test]
    fn console_log_run_call() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        let ConsoleLogTransaction {
            transaction,
            expected_call_data,
        } = deploy_console_log_contract(&mut fixture.provider_data)?;

        let pending_transaction = fixture
            .provider_data
            .sign_transaction_request(transaction)?;

        let result = fixture.provider_data.run_call(
            pending_transaction,
            &BlockSpec::latest(),
            &StateOverrides::default(),
        )?;

        let console_log_inputs = result.console_log_inputs;
        assert_eq!(console_log_inputs.len(), 1);
        assert_eq!(console_log_inputs[0], expected_call_data);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_empty() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let previous_block_number = fixture.provider_data.last_block_number();

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;
        assert!(result.block.transactions().is_empty());

        let current_block_number = fixture.provider_data.last_block_number();
        assert_eq!(current_block_number, previous_block_number + 1);

        let cached_state = fixture
            .provider_data
            .get_or_compute_state(result.block.header().number)?;

        let calculated_state = fixture.provider_data.blockchain.state_at_block_number(
            fixture.provider_data.last_block_number(),
            fixture.provider_data.irregular_state.state_overrides(),
        )?;

        assert_eq!(cached_state.state_root()?, calculated_state.state_root()?);

        Ok(())
    }

    #[test]
    fn mine_and_commit_blocks_empty() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        fixture
            .provider_data
            .mine_and_commit_blocks(1_000_000_000, 1)?;

        let cached_state = fixture
            .provider_data
            .get_or_compute_state(fixture.provider_data.last_block_number())?;

        let calculated_state = fixture.provider_data.blockchain.state_at_block_number(
            fixture.provider_data.last_block_number(),
            fixture.provider_data.irregular_state.state_overrides(),
        )?;

        assert_eq!(cached_state.state_root()?, calculated_state.state_root()?);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_single_transaction() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction = fixture.signed_dummy_transaction(0, None)?;
        let expected = *transaction.value();
        let receiver = transaction
            .kind()
            .to()
            .copied()
            .expect("Dummy transaction should have a receiver");

        fixture.provider_data.add_pending_transaction(transaction)?;

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        assert_eq!(result.block.transactions().len(), 1);

        let balance = fixture
            .provider_data
            .balance(receiver, Some(&BlockSpec::latest()))?;

        assert_eq!(balance, expected);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_two_transactions_different_senders() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction1 = fixture.signed_dummy_transaction(0, None)?;
        let transaction2 = fixture.signed_dummy_transaction(1, None)?;

        let receiver = transaction1
            .kind()
            .to()
            .copied()
            .expect("Dummy transaction should have a receiver");

        let expected = transaction1.value() + transaction2.value();

        fixture
            .provider_data
            .add_pending_transaction(transaction1)?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2)?;

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        assert_eq!(result.block.transactions().len(), 2);

        let balance = fixture
            .provider_data
            .balance(receiver, Some(&BlockSpec::latest()))?;

        assert_eq!(balance, expected);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_two_transactions_same_sender() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction1 = fixture.signed_dummy_transaction(0, Some(0))?;
        let transaction2 = fixture.signed_dummy_transaction(0, Some(1))?;

        let receiver = transaction1
            .kind()
            .to()
            .copied()
            .expect("Dummy transaction should have a receiver");

        let expected = transaction1.value() + transaction2.value();

        fixture
            .provider_data
            .add_pending_transaction(transaction1)?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2)?;

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        assert_eq!(result.block.transactions().len(), 2);

        let balance = fixture
            .provider_data
            .balance(receiver, Some(&BlockSpec::latest()))?;

        assert_eq!(balance, expected);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_removes_mined_transactions() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction = fixture.signed_dummy_transaction(0, None)?;

        fixture
            .provider_data
            .add_pending_transaction(transaction.clone())?;

        let num_pending_transactions = fixture.provider_data.pending_transactions().count();
        assert_eq!(num_pending_transactions, 1);

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        assert_eq!(result.block.transactions().len(), 1);

        let num_pending_transactions = fixture.provider_data.pending_transactions().count();
        assert_eq!(num_pending_transactions, 0);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_leaves_unmined_transactions() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        // SAFETY: literal is non-zero
        fixture
            .provider_data
            .set_block_gas_limit(unsafe { NonZeroU64::new_unchecked(55_000) })?;

        // Actual gas usage is 21_000
        let transaction1 = fixture.signed_dummy_transaction(0, Some(0))?;
        let transaction3 = fixture.signed_dummy_transaction(0, Some(1))?;

        // Too expensive to mine
        let transaction2 = {
            let request = fixture.dummy_transaction_request(1, 40_000, None)?;
            fixture.provider_data.sign_transaction_request(request)?
        };

        fixture
            .provider_data
            .add_pending_transaction(transaction1.clone())?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2.clone())?;
        fixture
            .provider_data
            .add_pending_transaction(transaction3.clone())?;

        let pending_transactions = fixture
            .provider_data
            .pending_transactions()
            .cloned()
            .collect::<Vec<_>>();

        assert!(pending_transactions.contains(&transaction1));
        assert!(pending_transactions.contains(&transaction2));
        assert!(pending_transactions.contains(&transaction3));

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        // Check that only the first and third transactions were mined
        assert_eq!(result.block.transactions().len(), 2);
        assert!(fixture
            .provider_data
            .transaction_receipt(transaction1.transaction_hash())?
            .is_some());
        assert!(fixture
            .provider_data
            .transaction_receipt(transaction3.transaction_hash())?
            .is_some());

        // Check that the second transaction is still pending
        let pending_transactions = fixture
            .provider_data
            .pending_transactions()
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(pending_transactions, vec![transaction2]);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_fifo_ordering() -> anyhow::Result<()> {
        let default_config = create_test_config();
        let config = ProviderConfig {
            mining: MiningConfig {
                mem_pool: MemPoolConfig {
                    order: MineOrdering::Fifo,
                },
                ..default_config.mining
            },
            ..default_config
        };

        let runtime = runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("provider-data-test")
            .build()?;

        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new(runtime, config)?;

        let transaction1 = fixture.signed_dummy_transaction(0, None)?;
        let transaction2 = fixture.signed_dummy_transaction(1, None)?;

        fixture
            .provider_data
            .add_pending_transaction(transaction1.clone())?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2.clone())?;

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        assert_eq!(result.block.transactions().len(), 2);

        let receipt1 = fixture
            .provider_data
            .transaction_receipt(transaction1.transaction_hash())?
            .expect("receipt should exist");

        assert_eq!(receipt1.transaction_index, 0);

        let receipt2 = fixture
            .provider_data
            .transaction_receipt(transaction2.transaction_hash())?
            .expect("receipt should exist");

        assert_eq!(receipt2.transaction_index, 1);

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_correct_gas_used() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction1 = fixture.signed_dummy_transaction(0, None)?;
        let transaction2 = fixture.signed_dummy_transaction(1, None)?;

        fixture
            .provider_data
            .add_pending_transaction(transaction1.clone())?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2.clone())?;

        let result = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        let receipt1 = fixture
            .provider_data
            .transaction_receipt(transaction1.transaction_hash())?
            .expect("receipt should exist");
        let receipt2 = fixture
            .provider_data
            .transaction_receipt(transaction2.transaction_hash())?
            .expect("receipt should exist");

        assert_eq!(receipt1.gas_used, 21_000);
        assert_eq!(receipt2.gas_used, 21_000);
        assert_eq!(
            result.block.header().gas_used,
            receipt1.gas_used + receipt2.gas_used
        );

        Ok(())
    }

    #[test]
    fn mine_and_commit_block_rewards_miner() -> anyhow::Result<()> {
        let default_config = create_test_config();
        let config = ProviderConfig {
            hardfork: l1::SpecId::BERLIN,
            ..default_config
        };

        let runtime = runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("provider-data-test")
            .build()?;

        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new(runtime, config)?;

        let miner = fixture.provider_data.beneficiary;
        let previous_miner_balance = fixture
            .provider_data
            .balance(miner, Some(&BlockSpec::latest()))?;

        let transaction = fixture.signed_dummy_transaction(0, None)?;
        fixture
            .provider_data
            .add_pending_transaction(transaction.clone())?;

        fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        let miner_balance = fixture
            .provider_data
            .balance(miner, Some(&BlockSpec::latest()))?;

        assert!(miner_balance > previous_miner_balance);

        Ok(())
    }

    #[test]
    fn mine_and_commit_blocks_increases_block_number() -> anyhow::Result<()> {
        const NUM_MINED_BLOCKS: u64 = 10;

        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let previous_block_number = fixture.provider_data.last_block_number();

        fixture
            .provider_data
            .mine_and_commit_blocks(NUM_MINED_BLOCKS, 1)?;

        assert_eq!(
            fixture.provider_data.last_block_number(),
            previous_block_number + NUM_MINED_BLOCKS
        );
        assert_eq!(
            fixture.provider_data.last_block()?.header().number,
            previous_block_number + NUM_MINED_BLOCKS
        );

        Ok(())
    }

    #[test]
    fn mine_and_commit_blocks_works_with_snapshots() -> anyhow::Result<()> {
        const NUM_MINED_BLOCKS: u64 = 10;

        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction1 = fixture.signed_dummy_transaction(0, None)?;
        let transaction2 = fixture.signed_dummy_transaction(1, None)?;

        let original_block_number = fixture.provider_data.last_block_number();

        fixture
            .provider_data
            .add_pending_transaction(transaction1.clone())?;

        let snapshot_id = fixture.provider_data.make_snapshot();
        assert_eq!(
            fixture.provider_data.last_block_number(),
            original_block_number
        );

        // Mine block after snapshot
        fixture
            .provider_data
            .mine_and_commit_blocks(NUM_MINED_BLOCKS, 1)?;

        assert_eq!(
            fixture.provider_data.last_block_number(),
            original_block_number + NUM_MINED_BLOCKS
        );

        let reverted = fixture.provider_data.revert_to_snapshot(snapshot_id);
        assert!(reverted);

        assert_eq!(
            fixture.provider_data.last_block_number(),
            original_block_number
        );

        fixture
            .provider_data
            .mine_and_commit_blocks(NUM_MINED_BLOCKS, 1)?;

        let block_number_before_snapshot = fixture.provider_data.last_block_number();

        // Mine block before snapshot
        let snapshot_id = fixture.provider_data.make_snapshot();

        fixture
            .provider_data
            .add_pending_transaction(transaction2.clone())?;

        fixture.provider_data.mine_and_commit_blocks(1, 1)?;

        let reverted = fixture.provider_data.revert_to_snapshot(snapshot_id);
        assert!(reverted);

        assert_eq!(
            fixture.provider_data.last_block_number(),
            block_number_before_snapshot
        );

        Ok(())
    }

    #[test]
    fn next_filter_id() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let mut prev_filter_id = fixture.provider_data.last_filter_id;
        for _ in 0..10 {
            let filter_id = fixture.provider_data.next_filter_id();
            assert!(prev_filter_id < filter_id);
            prev_filter_id = filter_id;
        }

        Ok(())
    }

    #[test]
    fn pending_transactions_returns_pending_and_queued() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local().unwrap();

        let transaction1 = fixture.signed_dummy_transaction(0, Some(0))?;
        fixture
            .provider_data
            .add_pending_transaction(transaction1.clone())?;

        let transaction2 = fixture.signed_dummy_transaction(0, Some(2))?;
        fixture
            .provider_data
            .add_pending_transaction(transaction2.clone())?;

        let transaction3 = fixture.signed_dummy_transaction(0, Some(3))?;
        fixture
            .provider_data
            .add_pending_transaction(transaction3.clone())?;

        let pending_transactions = fixture
            .provider_data
            .pending_transactions()
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(
            pending_transactions,
            vec![transaction1, transaction2, transaction3]
        );

        Ok(())
    }

    #[test]
    fn set_balance_updates_mem_pool() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction = fixture.impersonated_dummy_transaction()?;
        let transaction_hash = fixture.provider_data.add_pending_transaction(transaction)?;

        assert!(fixture
            .provider_data
            .mem_pool
            .transaction_by_hash(&transaction_hash)
            .is_some());

        fixture
            .provider_data
            .set_balance(fixture.impersonated_account, U256::from(100))?;

        assert!(fixture
            .provider_data
            .mem_pool
            .transaction_by_hash(&transaction_hash)
            .is_none());

        Ok(())
    }

    #[test]
    fn transaction_by_invalid_hash() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let non_existing_tx = fixture.provider_data.transaction_by_hash(&B256::ZERO)?;

        assert!(non_existing_tx.is_none());

        Ok(())
    }

    #[test]
    fn pending_transaction_by_hash() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction_request = fixture.signed_dummy_transaction(0, None)?;
        let transaction_hash = fixture
            .provider_data
            .add_pending_transaction(transaction_request)?;

        let transaction_result = fixture
            .provider_data
            .transaction_by_hash(&transaction_hash)?
            .context("transaction not found")?;

        assert_eq!(
            transaction_result.transaction.transaction_hash(),
            &transaction_hash
        );

        Ok(())
    }

    #[test]
    fn transaction_by_hash() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let transaction_request = fixture.signed_dummy_transaction(0, None)?;
        let transaction_hash = fixture
            .provider_data
            .add_pending_transaction(transaction_request)?;

        let results = fixture
            .provider_data
            .mine_and_commit_block(BlockOptions::default())?;

        // Make sure transaction was mined successfully.
        assert!(results
            .transaction_results
            .first()
            .context("failed to mine transaction")?
            .is_success());
        // Sanity check that the mempool is empty.
        assert_eq!(fixture.provider_data.mem_pool.transactions().count(), 0);

        let transaction_result = fixture
            .provider_data
            .transaction_by_hash(&transaction_hash)?
            .context("transaction not found")?;

        assert_eq!(
            transaction_result.transaction.transaction_hash(),
            &transaction_hash
        );

        Ok(())
    }

    #[test]
    fn sign_typed_data_v4() -> anyhow::Result<()> {
        let fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        // This test was taken from the `eth_signTypedData` example from the
        // EIP-712 specification via Hardhat.
        // <https://eips.ethereum.org/EIPS/eip-712#eth_signtypeddata>

        let address: Address = "0xCD2a3d9F938E13CD947Ec05AbC7FE734Df8DD826".parse()?;
        let message = json!({
          "types": {
            "EIP712Domain": [
              { "name": "name", "type": "string" },
              { "name": "version", "type": "string" },
              { "name": "chainId", "type": "uint256" },
              { "name": "verifyingContract", "type": "address" },
            ],
            "Person": [
              { "name": "name", "type": "string" },
              { "name": "wallet", "type": "address" },
            ],
            "Mail": [
              { "name": "from", "type": "Person" },
              { "name": "to", "type": "Person" },
              { "name": "contents", "type": "string" },
            ],
          },
          "primaryType": "Mail",
          "domain": {
            "name": "Ether Mail",
            "version": "1",
            "chainId": 1,
            "verifyingContract": "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC",
          },
          "message": {
            "from": {
              "name": "Cow",
              "wallet": "0xCD2a3d9F938E13CD947Ec05AbC7FE734Df8DD826",
            },
            "to": {
              "name": "Bob",
              "wallet": "0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB",
            },
            "contents": "Hello, Bob!",
          },
        });
        let message: TypedData = serde_json::from_value(message)?;

        let signature = fixture
            .provider_data
            .sign_typed_data_v4(&address, &message)?;

        let expected_signature = "0x4355c47d63924e8a72e509b65029052eb6c299d53a04e167c5775fd466751c9d07299936d304c153f6443dfa05f40ff007d72911b6f72307f996231605b915621c";

        assert_eq!(hex::decode(expected_signature)?, signature.to_vec(),);

        Ok(())
    }

    #[cfg(feature = "test-remote")]
    mod alchemy {
        use edr_eth::result::HaltReason;
        use edr_evm::impl_full_block_tests;
        use edr_test_utils::env::get_alchemy_url;

        use super::*;
        use crate::test_utils::FORK_BLOCK_NUMBER;

        #[test]
        fn reset_local_to_forking() -> anyhow::Result<()> {
            let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

            let fork_config = Some(ForkConfig {
                json_rpc_url: get_alchemy_url(),
                // Random recent block for better cache consistency
                block_number: Some(FORK_BLOCK_NUMBER),
                http_headers: None,
            });

            let block_spec = BlockSpec::Number(FORK_BLOCK_NUMBER);

            assert_eq!(fixture.provider_data.last_block_number(), 0);

            fixture.provider_data.reset(fork_config)?;

            // We're fetching a specific block instead of the last block number for the
            // forked blockchain, because the last block number query cannot be
            // cached.
            assert!(fixture
                .provider_data
                .block_by_block_spec(&block_spec)?
                .is_some());

            Ok(())
        }

        #[test]
        fn reset_forking_to_local() -> anyhow::Result<()> {
            let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_forked(None)?;

            // We're fetching a specific block instead of the last block number for the
            // forked blockchain, because the last block number query cannot be
            // cached.
            assert!(fixture
                .provider_data
                .block_by_block_spec(&BlockSpec::Number(FORK_BLOCK_NUMBER))?
                .is_some());

            fixture.provider_data.reset(None)?;

            assert_eq!(fixture.provider_data.last_block_number(), 0);

            Ok(())
        }

        #[test]
        fn run_call_in_hardfork_context() -> anyhow::Result<()> {
            use alloy_sol_types::{sol, SolCall};
            use edr_evm::transaction::TransactionError;
            use edr_rpc_eth::CallRequest;

            use crate::{
                requests::eth::resolve_call_request, test_utils::create_test_config_with_fork,
            };

            sol! { function Hello() public pure returns (string); }

            fn assert_decoded_output(result: ExecutionResult<HaltReason>) -> anyhow::Result<()> {
                let output = result.into_output().expect("Call must have output");
                let decoded = HelloCall::abi_decode_returns(output.as_ref(), false)?;

                assert_eq!(decoded._0, "Hello World");
                Ok(())
            }

            /// Executes a call to method `Hello` on contract `HelloWorld`,
            /// deployed to mainnet.
            ///
            /// Should return a string `"Hello World"`.
            fn call_hello_world_contract(
                data: &mut ProviderData<L1ChainSpec>,
                block_spec: BlockSpec,
                request: CallRequest,
            ) -> Result<CallResult<HaltReason>, ProviderError<L1ChainSpec>> {
                let state_overrides = StateOverrides::default();

                let transaction =
                    resolve_call_request(data, request, &block_spec, &state_overrides)?;

                data.run_call(transaction, &block_spec, &state_overrides)
            }

            const EIP_1559_ACTIVATION_BLOCK: u64 = 12_965_000;
            const HELLO_WORLD_CONTRACT_ADDRESS: &str = "0xe36613A299bA695aBA8D0c0011FCe95e681f6dD3";

            let hello_world_contract_address: Address = HELLO_WORLD_CONTRACT_ADDRESS.parse()?;
            let hello_world_contract_call = HelloCall::new(());

            let runtime = runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .thread_name("provider-data-test")
                .build()?;

            let default_config = create_test_config_with_fork(Some(ForkConfig {
                json_rpc_url: get_alchemy_url(),
                block_number: Some(EIP_1559_ACTIVATION_BLOCK),
                http_headers: None,
            }));

            let config = ProviderConfig {
                // SAFETY: literal is non-zero
                block_gas_limit: unsafe { NonZeroU64::new_unchecked(1_000_000) },
                chain_id: 1,
                coinbase: Address::ZERO,
                hardfork: l1::SpecId::LONDON,
                network_id: 1,
                ..default_config
            };

            let mut fixture = ProviderTestFixture::<L1ChainSpec>::new(runtime, config)?;

            let default_call = CallRequest {
                from: Some(fixture.nth_local_account(0)?),
                to: Some(hello_world_contract_address),
                gas: Some(1_000_000),
                value: Some(U256::ZERO),
                data: Some(hello_world_contract_call.abi_encode().into()),
                ..CallRequest::default()
            };

            // Should accept post-EIP-1559 gas semantics when running in the context of a
            // post-EIP-1559 block
            let result = call_hello_world_contract(
                &mut fixture.provider_data,
                BlockSpec::Number(EIP_1559_ACTIVATION_BLOCK),
                CallRequest {
                    max_fee_per_gas: Some(U256::ZERO),
                    ..default_call.clone()
                },
            )?;

            assert_decoded_output(result.execution_result)?;

            // Should accept pre-EIP-1559 gas semantics when running in the context of a
            // pre-EIP-1559 block
            let result = call_hello_world_contract(
                &mut fixture.provider_data,
                BlockSpec::Number(EIP_1559_ACTIVATION_BLOCK - 1),
                CallRequest {
                    gas_price: Some(U256::ZERO),
                    ..default_call.clone()
                },
            )?;

            assert_decoded_output(result.execution_result)?;

            // Should throw when given post-EIP-1559 gas semantics and when running in the
            // context of a pre-EIP-1559 block
            let result = call_hello_world_contract(
                &mut fixture.provider_data,
                BlockSpec::Number(EIP_1559_ACTIVATION_BLOCK - 1),
                CallRequest {
                    max_fee_per_gas: Some(U256::ZERO),
                    ..default_call.clone()
                },
            );

            assert!(matches!(
                result,
                Err(ProviderError::RunTransaction(
                    TransactionError::Eip1559Unsupported
                ))
            ));

            // Should accept pre-EIP-1559 gas semantics when running in the context of a
            // post-EIP-1559 block
            let result = call_hello_world_contract(
                &mut fixture.provider_data,
                BlockSpec::Number(EIP_1559_ACTIVATION_BLOCK),
                CallRequest {
                    gas_price: Some(U256::ZERO),
                    ..default_call.clone()
                },
            )?;

            assert_decoded_output(result.execution_result)?;

            // Should support a historical call in the context of a block added via
            // `mine_and_commit_blocks`
            let previous_block_number = fixture.provider_data.last_block_number();

            fixture.provider_data.mine_and_commit_blocks(100, 1)?;

            let result = call_hello_world_contract(
                &mut fixture.provider_data,
                BlockSpec::Number(previous_block_number + 50),
                CallRequest {
                    max_fee_per_gas: Some(U256::ZERO),
                    ..default_call
                },
            )?;

            assert_decoded_output(result.execution_result)?;

            Ok(())
        }

        impl_full_block_tests! {
            mainnet_byzantium => L1ChainSpec {
                block_number: 4_370_001,
                url: get_alchemy_url(),
            },
            mainnet_constantinople => L1ChainSpec {
                block_number: 7_280_001,
                url: get_alchemy_url(),
            },
            mainnet_istanbul => L1ChainSpec {
                block_number: 9_069_001,
                url: get_alchemy_url(),
            },
            mainnet_muir_glacier => L1ChainSpec {
                block_number: 9_300_077,
                url: get_alchemy_url(),
            },
            mainnet_shanghai => L1ChainSpec {
                block_number: 17_050_001,
                url: get_alchemy_url(),
            },
            // This block contains a sequence of transaction that first raise
            // an empty account's balance and then decrease it
            mainnet_19318016 => L1ChainSpec {
                block_number: 19_318_016,
                url: get_alchemy_url(),
            },
            // This block has both EIP-2930 and EIP-1559 transactions
            sepolia_eip_1559_2930 => L1ChainSpec {
                block_number: 5_632_795,
                url: get_alchemy_url().replace("mainnet", "sepolia"),
            },
            sepolia_shanghai => L1ChainSpec {
                block_number: 3_095_000,
                url: get_alchemy_url().replace("mainnet", "sepolia"),
            },
            // This block has an EIP-4844 transaction
            mainnet_cancun => L1ChainSpec {
                block_number: 19_529_021,
                url: get_alchemy_url(),
            },
            // This block contains a transaction that uses the KZG point evaluation
            // precompile, introduced in Cancun
            mainnet_cancun2 => L1ChainSpec {
                block_number: 19_562_047,
                url: get_alchemy_url(),
            },
        }
    }
}
