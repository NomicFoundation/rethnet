use std::sync::Arc;

use edr_eth::{result::InvalidTransaction, transaction::TransactionValidation};
use tokio::{runtime, sync::Mutex};

use crate::{
    data::ProviderData, interval::IntervalMiner, requests, spec::SyncProviderSpec,
    time::TimeSinceEpoch, IntervalConfig, ProviderError,
};

pub fn handle_set_interval_mining<
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
    data: Arc<Mutex<ProviderData<ChainSpecT, TimerT>>>,
    interval_miner: &mut Option<IntervalMiner<ChainSpecT, TimerT>>,
    runtime: runtime::Handle,
    config: requests::IntervalConfig,
) -> Result<bool, ProviderError<ChainSpecT>> {
    let config: Option<IntervalConfig> = config.try_into()?;
    *interval_miner = config.map(|config| IntervalMiner::new(runtime, config, data.clone()));

    Ok(true)
}
