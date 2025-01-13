use edr_eth::{Address, Bytes, U256};

use crate::{data::ProviderData, spec::SyncProviderSpec, time::TimeSinceEpoch, ProviderError};

pub fn handle_set_balance<ChainSpecT: SyncProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    address: Address,
    balance: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_balance(address, balance)?;

    Ok(true)
}

pub fn handle_set_code<ChainSpecT: SyncProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    address: Address,
    code: Bytes,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_code(address, code)?;

    Ok(true)
}

pub fn handle_set_nonce<ChainSpecT: SyncProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    address: Address,
    nonce: u64,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_nonce(address, nonce)?;

    Ok(true)
}

pub fn handle_set_storage_at<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    address: Address,
    index: U256,
    value: U256,
) -> Result<bool, ProviderError<ChainSpecT>> {
    data.set_account_storage_slot(address, index, value)?;

    Ok(true)
}
