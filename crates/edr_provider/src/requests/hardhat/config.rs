use edr_eth::{Address, B256, U256};
use edr_evm::Block as _;

use crate::{
    data::ProviderData,
    requests::{eth::client_version, hardhat::rpc_types::Metadata},
    spec::{ProviderSpec, SyncProviderSpec},
    time::TimeSinceEpoch,
    ProviderError,
};

pub fn handle_get_automine_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<bool, ProviderError<ChainSpecT>> {
    Ok(data.is_auto_mining())
}

pub fn handle_metadata_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<Metadata, ProviderError<ChainSpecT>> {
    Ok(Metadata {
        client_version: client_version(),
        chain_id: data.chain_id(),
        instance_id: *data.instance_id(),
        latest_block_number: data.last_block_number(),
        latest_block_hash: *data.last_block()?.block_hash(),
        forked_network: data.fork_metadata().cloned(),
    })
}

pub fn handle_set_coinbase_request<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    coinbase: Address,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_coinbase(coinbase);

    Ok(true)
}

pub fn handle_set_min_gas_price<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    min_gas_price: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_min_gas_price(min_gas_price)?;

    Ok(true)
}

pub fn handle_set_next_block_base_fee_per_gas_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    base_fee_per_gas: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_next_block_base_fee_per_gas(base_fee_per_gas)?;

    Ok(true)
}

pub fn handle_set_prev_randao_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    prev_randao: B256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_next_prev_randao(prev_randao)?;

    Ok(true)
}
