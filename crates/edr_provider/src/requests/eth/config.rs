use edr_eth::{Address, U256, U64};
use edr_evm::spec::RuntimeSpec;

use crate::{
    data::ProviderData,
    spec::{ProviderSpec, SyncProviderSpec},
    time::TimeSinceEpoch,
    ProviderError,
};

pub fn handle_blob_base_fee<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    let base_fee = data.next_block_base_fee_per_blob_gas()?.unwrap_or_default();

    Ok(base_fee)
}

pub fn handle_gas_price<ChainSpecT: SyncProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    data.gas_price()
}

pub fn handle_coinbase_request<ChainSpecT: ProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<Address, ProviderError<ChainSpecT>> {
    Ok(data.coinbase())
}

pub fn handle_max_priority_fee_per_gas<ChainSpecT: RuntimeSpec>(
) -> Result<U256, ProviderError<ChainSpecT>> {
    // 1 gwei
    Ok(U256::from(1_000_000_000))
}

pub fn handle_mining<ChainSpecT: RuntimeSpec>() -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(false)
}

pub fn handle_net_listening_request<ChainSpecT: RuntimeSpec>(
) -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(true)
}

pub fn handle_net_peer_count_request<ChainSpecT: RuntimeSpec>(
) -> Result<U64, ProviderError<ChainSpecT>> {
    Ok(U64::from(0))
}

pub fn handle_net_version_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<String, ProviderError<ChainSpecT>> {
    Ok(data.network_id())
}

pub fn handle_syncing<ChainSpecT: RuntimeSpec>() -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(false)
}
