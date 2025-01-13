use edr_eth::{
    result::InvalidTransaction, transaction::TransactionValidation, Address, BlockSpec, U256, U64,
};

use crate::{
    data::ProviderData, requests::validation::validate_post_merge_block_tags,
    spec::SyncProviderSpec, time::TimeSinceEpoch, ProviderError,
};

pub fn handle_block_number_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<U64, ProviderError<ChainSpecT>> {
    Ok(U64::from(data.last_block_number()))
}

pub fn handle_chain_id_request<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<U64, ProviderError<ChainSpecT>> {
    Ok(U64::from(data.chain_id()))
}

pub fn handle_get_transaction_count_request<
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
    data: &mut ProviderData<ChainSpecT, TimerT>,
    address: Address,
    block_spec: Option<BlockSpec>,
) -> Result<U256, ProviderError<ChainSpecT>> {
    if let Some(block_spec) = block_spec.as_ref() {
        validate_post_merge_block_tags(data.hardfork(), block_spec)?;
    }

    data.get_transaction_count(address, block_spec.as_ref())
        .map(U256::from)
}
