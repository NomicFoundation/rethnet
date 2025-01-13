use edr_eth::Address;

use crate::{data::ProviderData, spec::ProviderSpec, time::TimeSinceEpoch, ProviderError};

/// `require_canonical`: whether the server should additionally raise a JSON-RPC
/// error if the block is not in the canonical chain
pub fn handle_accounts_request<ChainSpecT: ProviderSpec<TimerT>, TimerT: Clone + TimeSinceEpoch>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<Vec<Address>, ProviderError<ChainSpecT>> {
    Ok(data.accounts().copied().collect())
}
