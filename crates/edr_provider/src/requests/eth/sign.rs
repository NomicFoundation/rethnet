use alloy_dyn_abi::eip712::TypedData;
use edr_eth::{Address, Bytes};

use crate::{data::ProviderData, spec::ProviderSpec, time::TimeSinceEpoch, ProviderError};

pub fn handle_sign_request<ChainSpecT: ProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &ProviderData<ChainSpecT, TimerT>,
    message: Bytes,
    address: Address,
) -> Result<Bytes, ProviderError<ChainSpecT>> {
    Ok((&data.sign(&address, message)?).into())
}

pub fn handle_sign_typed_data_v4<
    ChainSpecT: ProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    address: Address,
    message: TypedData,
) -> Result<Bytes, ProviderError<ChainSpecT>> {
    Ok((&data.sign_typed_data_v4(&address, &message)?).into())
}
