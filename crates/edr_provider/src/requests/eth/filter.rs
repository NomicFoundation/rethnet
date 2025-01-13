use core::iter;

use edr_eth::{
    filter::{FilteredEvents, LogFilterOptions, LogOutput, OneOrMore, SubscriptionType},
    BlockSpec, BlockTag, Eip1898BlockSpec, HashSet, U256,
};
use edr_evm::Block as _;

use crate::{
    data::ProviderData,
    filter::LogFilter,
    requests::validation::validate_post_merge_block_tags,
    spec::{ProviderSpec, SyncProviderSpec},
    time::TimeSinceEpoch,
    ProviderError,
};

pub fn handle_get_filter_changes_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    filter_id: U256,
) -> Result<Option<FilteredEvents>, ProviderError<ChainSpecT>> {
    Ok(data.get_filter_changes(&filter_id))
}

pub fn handle_get_filter_logs_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    filter_id: U256,
) -> Result<Option<Vec<LogOutput>>, ProviderError<ChainSpecT>> {
    data.get_filter_logs(&filter_id)
}

pub fn handle_get_logs_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    filter_options: LogFilterOptions,
) -> Result<Vec<LogOutput>, ProviderError<ChainSpecT>> {
    let hardfork = data.hardfork();
    // Hardhat integration tests expect validation in this order.
    if let Some(from_block) = &filter_options.from_block {
        validate_post_merge_block_tags(hardfork, from_block)?;
    }
    if let Some(to_block) = &filter_options.to_block {
        validate_post_merge_block_tags(hardfork, to_block)?;
    }

    let filter = validate_filter_criteria::<true, ChainSpecT, TimerT>(data, filter_options)?;
    data.logs(filter)
        .map(|logs| logs.iter().map(LogOutput::from).collect())
}

pub fn handle_new_block_filter_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    data.add_block_filter::<false>()
}

pub fn handle_new_log_filter_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    filter_criteria: LogFilterOptions,
) -> Result<U256, ProviderError<ChainSpecT>> {
    let filter_criteria =
        validate_filter_criteria::<false, ChainSpecT, TimerT>(data, filter_criteria)?;
    data.add_log_filter::<false>(filter_criteria)
}

pub fn handle_new_pending_transaction_filter_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    Ok(data.add_pending_transaction_filter::<false>())
}

pub fn handle_subscribe_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    subscription_type: SubscriptionType,
    filter_criteria: Option<LogFilterOptions>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    match subscription_type {
        SubscriptionType::Logs => {
            let filter_criteria = filter_criteria.ok_or_else(|| {
                ProviderError::InvalidArgument("Missing params argument".to_string())
            })?;
            let filter_criteria =
                validate_filter_criteria::<false, ChainSpecT, TimerT>(data, filter_criteria)?;
            data.add_log_filter::<true>(filter_criteria)
        }
        SubscriptionType::NewHeads => data.add_block_filter::<true>(),
        SubscriptionType::NewPendingTransactions => {
            Ok(data.add_pending_transaction_filter::<true>())
        }
    }
}

pub fn handle_uninstall_filter_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    filter_id: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(data.remove_filter(&filter_id))
}

pub fn handle_unsubscribe_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    filter_id: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(data.remove_subscription(&filter_id))
}

fn validate_filter_criteria<
    const SHOULD_RESOLVE_LATEST: bool,
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    filter: LogFilterOptions,
) -> Result<LogFilter, ProviderError<ChainSpecT>> {
    fn normalize_block_spec<
        ChainSpecT: SyncProviderSpec<TimerT>,
        TimerT: Clone + TimeSinceEpoch,
    >(
        data: &ProviderData<ChainSpecT, TimerT>,
        block_spec: Option<BlockSpec>,
    ) -> Result<Option<u64>, ProviderError<ChainSpecT>> {
        if let Some(block_spec) = &block_spec {
            validate_post_merge_block_tags(data.hardfork(), block_spec)?;
        }

        let block_number = match block_spec {
            Some(
                BlockSpec::Number(block_number)
                | BlockSpec::Eip1898(Eip1898BlockSpec::Number { block_number }),
            ) => Some(block_number),
            Some(BlockSpec::Tag(BlockTag::Earliest)) => Some(0),
            Some(BlockSpec::Tag(
                BlockTag::Latest | BlockTag::Pending | BlockTag::Safe | BlockTag::Finalized,
            ))
            | None => None,
            Some(BlockSpec::Eip1898(Eip1898BlockSpec::Hash { block_hash, .. })) => {
                let block =
                    data.block_by_hash(&block_hash)?
                        .ok_or(ProviderError::InvalidArgument(
                            "blockHash cannot be found".to_string(),
                        ))?;
                Some(block.header().number)
            }
        };

        Ok(block_number)
    }

    let (from_block, to_block) = if let Some(block_hash) = filter.block_hash {
        if filter.from_block.is_some() || filter.to_block.is_some() {
            return Err(ProviderError::InvalidArgument(
                "blockHash is mutually exclusive with fromBlock/toBlock".to_string(),
            ));
        }

        let block = data.block_by_hash(&block_hash)?.ok_or_else(|| {
            ProviderError::InvalidArgument("blockHash cannot be found".to_string())
        })?;

        let block_number = block.header().number;

        (block_number, Some(block_number))
    } else {
        let from_block = normalize_block_spec(data, filter.from_block)?
            .unwrap_or_else(|| data.last_block_number());

        let mut to_block = normalize_block_spec(data, filter.to_block)?;
        if SHOULD_RESOLVE_LATEST && to_block.is_none() {
            to_block = Some(data.last_block_number());
        }

        (from_block, to_block)
    };

    let addresses = filter
        .address
        .map_or(HashSet::new(), |addresses| match addresses {
            OneOrMore::One(address) => iter::once(address).collect(),
            OneOrMore::Many(addresses) => addresses.into_iter().collect(),
        });

    let normalized_topics = filter.topics.map_or(Vec::new(), |topics| {
        topics
            .into_iter()
            .map(|topics| {
                topics.map(|topics| match topics {
                    OneOrMore::One(topic) => vec![topic],
                    OneOrMore::Many(topics) => topics,
                })
            })
            .collect()
    });

    Ok(LogFilter {
        from_block,
        to_block,
        addresses,
        normalized_topics,
    })
}
