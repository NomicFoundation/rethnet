use std::sync::Arc;

use derive_where::derive_where;
use dyn_clone::DynClone;
use edr_eth::{filter::LogOutput, B256, U256};
use edr_evm::{spec::RuntimeSpec, BlockAndTotalDifficulty};

/// Subscription event.
#[derive_where(Clone, Debug)]
pub struct SubscriptionEvent<ChainSpecT: RuntimeSpec> {
    pub filter_id: U256,
    pub result: SubscriptionEventData<ChainSpecT>,
}

/// Subscription event data.
#[derive_where(Clone, Debug)]
pub enum SubscriptionEventData<ChainSpecT: RuntimeSpec> {
    Logs(Vec<LogOutput>),
    NewHeads(BlockAndTotalDifficulty<Arc<ChainSpecT::Block>, ChainSpecT::SignedTransaction>),
    NewPendingTransactions(B256),
}

/// Supertrait for subscription callbacks.
pub trait SyncSubscriberCallback<ChainSpecT: RuntimeSpec>:
    Fn(SubscriptionEvent<ChainSpecT>) + DynClone + Send + Sync
{
}

impl<ChainSpecT: RuntimeSpec, F> SyncSubscriberCallback<ChainSpecT> for F where
    F: Fn(SubscriptionEvent<ChainSpecT>) + DynClone + Send + Sync
{
}

dyn_clone::clone_trait_object!(<ChainSpecT> SyncSubscriberCallback<ChainSpecT> where ChainSpecT: RuntimeSpec);
