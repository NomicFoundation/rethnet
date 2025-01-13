use edr_eth::{
    fee_history::FeeHistoryResult,
    l1,
    result::InvalidTransaction,
    reward_percentile::RewardPercentile,
    transaction::{signed::FakeSign as _, TransactionMut, TransactionValidation},
    BlockSpec, U256, U64,
};
use edr_evm::{state::StateOverrides, transaction, Block as _};

use crate::{
    data::ProviderData,
    requests::validation::validate_post_merge_block_tags,
    spec::{CallContext, FromRpcType as _, MaybeSender as _, SyncProviderSpec},
    time::TimeSinceEpoch,
    ProviderError, ProviderResultWithTraces,
};

pub fn handle_estimate_gas<
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
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    request: ChainSpecT::RpcCallRequest,
    block_spec: Option<BlockSpec>,
) -> ProviderResultWithTraces<U64, ChainSpecT> {
    // Matching Hardhat behavior in defaulting to "pending" instead of "latest" for
    // estimate gas.
    let block_spec = block_spec.unwrap_or_else(BlockSpec::pending);

    let hardfork = data.hardfork();

    let transaction =
        resolve_estimate_gas_request(data, request, &block_spec, &StateOverrides::default())?;

    let result = data.estimate_gas(transaction.clone(), &block_spec);
    if let Err(ProviderError::EstimateGasTransactionFailure(failure)) = result {
        data.logger_mut()
            .log_estimate_gas_failure(hardfork, &transaction, &failure)
            .map_err(ProviderError::Logger)?;

        Err(ProviderError::TransactionFailed(
            failure.transaction_failure,
        ))
    } else {
        let result = result?;
        Ok((U64::from(result.estimation), result.traces))
    }
}

pub fn handle_fee_history<
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
    block_count: U256,
    newest_block: BlockSpec,
    reward_percentiles: Option<Vec<f64>>,
) -> Result<FeeHistoryResult, ProviderError<ChainSpecT>> {
    if data.evm_spec_id() < l1::SpecId::LONDON {
        return Err(ProviderError::InvalidInput(
            "eth_feeHistory is disabled. It only works with the London hardfork or a later one."
                .into(),
        ));
    }

    let block_count: u64 = block_count
        .try_into()
        .map_err(|_err| ProviderError::InvalidInput("blockCount should be at most 1024".into()))?;
    if block_count == 0 {
        return Err(ProviderError::InvalidInput(
            "blockCount should be at least 1".into(),
        ));
    }
    if block_count > 1024 {
        return Err(ProviderError::InvalidInput(
            "blockCount should be at most 1024".into(),
        ));
    }

    validate_post_merge_block_tags(data.hardfork(), &newest_block)?;

    let reward_percentiles = reward_percentiles.map(|percentiles| {
        let mut validated_percentiles = Vec::with_capacity(percentiles.len());
        for (i, percentile) in percentiles.iter().copied().enumerate() {
            validated_percentiles.push(RewardPercentile::try_from(percentile).map_err(|_err| {
                ProviderError::InvalidInput(format!(
                    "The reward percentile number {} is invalid. It must be a float between 0 and 100, but is {} instead.",
                    i + 1,
                    percentile
                ))
            })?);
            if i > 0 {
                let prev = percentiles[i - 1];
                if prev > percentile {
                    return Err(ProviderError::InvalidInput(format!("\
The reward percentiles should be in non-decreasing order, but the percentile number {i} is greater than the next one")));
                }
            }
        }
        Ok(validated_percentiles)
    }).transpose()?;

    data.fee_history(block_count, &newest_block, reward_percentiles)
}

fn resolve_estimate_gas_request<
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
    request: ChainSpecT::RpcCallRequest,
    block_spec: &BlockSpec,
    state_overrides: &StateOverrides,
) -> Result<ChainSpecT::SignedTransaction, ProviderError<ChainSpecT>> {
    let sender = request
        .maybe_sender()
        .copied()
        .unwrap_or_else(|| data.default_caller());

    let context = CallContext {
        data,
        block_spec,
        state_overrides,
        default_gas_price_fn: ProviderData::gas_price,
        max_fees_fn: |data, block_spec, max_fee_per_gas, max_priority_fee_per_gas| {
            let max_priority_fee_per_gas = max_priority_fee_per_gas.unwrap_or_else(|| {
                const DEFAULT: u64 = 1_000_000_000;
                let default = U256::from(DEFAULT);

                if let Some(max_fee_per_gas) = max_fee_per_gas {
                    default.min(max_fee_per_gas)
                } else {
                    default
                }
            });

            let max_fee_per_gas = max_fee_per_gas.map_or_else(
                || -> Result<U256, ProviderError<ChainSpecT>> {
                    let base_fee = if let Some(block) = data.block_by_block_spec(block_spec)? {
                        max_priority_fee_per_gas
                            + block.header().base_fee_per_gas.unwrap_or(U256::ZERO)
                    } else {
                        // Pending block
                        let base_fee = data
                            .next_block_base_fee_per_gas()?
                            .expect("This function can only be called for post-EIP-1559 blocks");

                        U256::from(2) * base_fee + max_priority_fee_per_gas
                    };

                    Ok(base_fee)
                },
                Ok,
            )?;

            Ok((max_fee_per_gas, max_priority_fee_per_gas))
        },
    };

    let request = ChainSpecT::TransactionRequest::from_rpc_type(request, context)?;
    let transaction = request.fake_sign(sender);

    transaction::validate(transaction, l1::SpecId::LATEST)
        .map_err(ProviderError::TransactionCreationError)
}

#[cfg(test)]
mod tests {
    use edr_eth::{transaction::Transaction as _, BlockTag};
    use edr_rpc_eth::CallRequest;
    use l1::L1ChainSpec;

    use super::*;
    use crate::test_utils::{pending_base_fee, ProviderTestFixture};

    #[test]
    fn resolve_estimate_gas_request_with_default_max_priority_fee() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let max_fee_per_gas =
            pending_base_fee(&mut fixture.provider_data)?.max(U256::from(10_000_000_000u64));

        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            max_fee_per_gas: Some(max_fee_per_gas),
            ..CallRequest::default()
        };

        let resolved = resolve_estimate_gas_request(
            &mut fixture.provider_data,
            request,
            &BlockSpec::pending(),
            &StateOverrides::default(),
        )?;

        assert_eq!(*resolved.gas_price(), max_fee_per_gas);
        assert_eq!(
            resolved.max_priority_fee_per_gas().cloned(),
            Some(U256::from(1_000_000_000u64))
        );

        Ok(())
    }

    #[test]
    fn resolve_estimate_gas_request_with_default_max_fee_when_pending_block() -> anyhow::Result<()>
    {
        let base_fee = U256::from(10);
        let max_priority_fee_per_gas = U256::from(1);

        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        fixture
            .provider_data
            .set_next_block_base_fee_per_gas(base_fee)?;

        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas),
            ..CallRequest::default()
        };

        let resolved = resolve_estimate_gas_request(
            &mut fixture.provider_data,
            request,
            &BlockSpec::pending(),
            &StateOverrides::default(),
        )?;

        assert_eq!(
            *resolved.gas_price(),
            U256::from(2) * base_fee + max_priority_fee_per_gas
        );
        assert_eq!(
            resolved.max_priority_fee_per_gas().cloned(),
            Some(max_priority_fee_per_gas)
        );

        Ok(())
    }

    #[test]
    fn resolve_estimate_gas_request_with_default_max_fee_when_historic_block() -> anyhow::Result<()>
    {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        fixture
            .provider_data
            .set_next_block_base_fee_per_gas(U256::from(10))?;

        let transaction = fixture.signed_dummy_transaction(0, None)?;
        fixture.provider_data.send_transaction(transaction)?;

        let last_block = fixture.provider_data.last_block()?;
        assert_eq!(last_block.header().number, 1);

        let max_priority_fee_per_gas = U256::from(1);
        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas),
            ..CallRequest::default()
        };

        let resolved = resolve_estimate_gas_request(
            &mut fixture.provider_data,
            request,
            &BlockSpec::Tag(BlockTag::Latest),
            &StateOverrides::default(),
        )?;

        assert_eq!(
            Some(*resolved.gas_price()),
            last_block
                .header()
                .base_fee_per_gas
                .map(|base_fee| base_fee + max_priority_fee_per_gas)
        );
        assert_eq!(
            resolved.max_priority_fee_per_gas().cloned(),
            Some(max_priority_fee_per_gas)
        );

        Ok(())
    }

    #[test]
    fn resolve_estimate_gas_request_with_capped_max_priority_fee() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;
        fixture
            .provider_data
            .set_next_block_base_fee_per_gas(U256::ZERO)?;

        let max_fee_per_gas = U256::from(123);

        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            max_fee_per_gas: Some(max_fee_per_gas),
            ..CallRequest::default()
        };

        let resolved = resolve_estimate_gas_request(
            &mut fixture.provider_data,
            request,
            &BlockSpec::pending(),
            &StateOverrides::default(),
        )?;

        assert_eq!(*resolved.gas_price(), max_fee_per_gas);
        assert_eq!(
            resolved.max_priority_fee_per_gas().cloned(),
            Some(max_fee_per_gas)
        );

        Ok(())
    }
}
