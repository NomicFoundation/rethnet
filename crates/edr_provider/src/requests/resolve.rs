use edr_eth::{
    l1::{self, L1ChainSpec},
    Bytes, U256,
};
use edr_evm::{blockchain::BlockchainErrorForChainSpec, transaction};
use edr_rpc_eth::{CallRequest, TransactionRequest};

use super::validation::validate_call_request;
use crate::{
    data::ProviderData,
    requests::validation::validate_send_transaction_request,
    spec::{CallContext, FromRpcType, TransactionContext},
    time::TimeSinceEpoch,
    ProviderError,
};

impl<TimerT: Clone + TimeSinceEpoch> FromRpcType<CallRequest, TimerT> for transaction::Request {
    type Context<'context> = CallContext<'context, L1ChainSpec, TimerT>;

    type Error = ProviderError<L1ChainSpec>;

    fn from_rpc_type(
        value: CallRequest,
        context: Self::Context<'_>,
    ) -> Result<transaction::Request, ProviderError<L1ChainSpec>> {
        let CallContext {
            data,
            block_spec,
            state_overrides,
            default_gas_price_fn,
            max_fees_fn,
        } = context;

        validate_call_request(data.evm_spec_id(), &value, block_spec)?;

        let CallRequest {
            from,
            to,
            gas,
            gas_price,
            max_fee_per_gas,
            max_priority_fee_per_gas,
            value,
            data: input,
            access_list,
            ..
        } = value;

        let chain_id = data.chain_id_at_block_spec(block_spec)?;
        let sender = from.unwrap_or_else(|| data.default_caller());
        let gas_limit = gas.unwrap_or_else(|| data.block_gas_limit());
        let input = input.map_or(Bytes::new(), Bytes::from);
        let nonce = data.nonce(&sender, Some(block_spec), state_overrides)?;
        let value = value.unwrap_or(U256::ZERO);

        let evm_spec_id = data.evm_spec_id();
        let request = if evm_spec_id < l1::SpecId::LONDON || gas_price.is_some() {
            let gas_price = gas_price.map_or_else(|| default_gas_price_fn(data), Ok)?;
            match access_list {
                Some(access_list) if evm_spec_id >= l1::SpecId::BERLIN => {
                    transaction::Request::Eip2930(transaction::request::Eip2930 {
                        nonce,
                        gas_price,
                        gas_limit,
                        value,
                        input,
                        kind: to.into(),
                        chain_id,
                        access_list,
                    })
                }
                _ => transaction::Request::Eip155(transaction::request::Eip155 {
                    nonce,
                    gas_price,
                    gas_limit,
                    kind: to.into(),
                    value,
                    input,
                    chain_id,
                }),
            }
        } else {
            let (max_fee_per_gas, max_priority_fee_per_gas) =
                max_fees_fn(data, block_spec, max_fee_per_gas, max_priority_fee_per_gas)?;

            transaction::Request::Eip1559(transaction::request::Eip1559 {
                chain_id,
                nonce,
                max_fee_per_gas,
                max_priority_fee_per_gas,
                gas_limit,
                kind: to.into(),
                value,
                input,
                access_list: access_list.unwrap_or_default(),
            })
        };

        Ok(request)
    }
}

impl<TimerT: Clone + TimeSinceEpoch> FromRpcType<TransactionRequest, TimerT>
    for transaction::Request
{
    type Context<'context> = TransactionContext<'context, L1ChainSpec, TimerT>;

    type Error = ProviderError<L1ChainSpec>;

    fn from_rpc_type(
        value: TransactionRequest,
        context: Self::Context<'_>,
    ) -> Result<transaction::Request, ProviderError<L1ChainSpec>> {
        const DEFAULT_MAX_PRIORITY_FEE_PER_GAS: u64 = 1_000_000_000;

        /// # Panics
        ///
        /// Panics if `data.evm_spec_id()` is less than `SpecId::LONDON`.
        fn calculate_max_fee_per_gas<TimerT: Clone + TimeSinceEpoch>(
            data: &ProviderData<L1ChainSpec, TimerT>,
            max_priority_fee_per_gas: U256,
        ) -> Result<U256, BlockchainErrorForChainSpec<L1ChainSpec>> {
            let base_fee_per_gas = data
                .next_block_base_fee_per_gas()?
                .expect("We already validated that the block is post-London.");
            Ok(U256::from(2) * base_fee_per_gas + max_priority_fee_per_gas)
        }

        let TransactionContext { data } = context;

        validate_send_transaction_request(data, &value)?;

        let TransactionRequest {
            from,
            to,
            gas_price,
            max_fee_per_gas,
            max_priority_fee_per_gas,
            gas,
            value,
            data: input,
            nonce,
            chain_id,
            access_list,
            // We ignore the transaction type
            transaction_type: _transaction_type,
            blobs: _blobs,
            blob_hashes: _blob_hashes,
        } = value;

        let chain_id = chain_id.unwrap_or_else(|| data.chain_id());
        let gas_limit = gas.unwrap_or_else(|| data.block_gas_limit());
        let input = input.map_or(Bytes::new(), Into::into);
        let nonce = nonce.map_or_else(|| data.account_next_nonce(&from), Ok)?;
        let value = value.unwrap_or(U256::ZERO);

        let request = match (
            gas_price,
            max_fee_per_gas,
            max_priority_fee_per_gas,
            access_list,
        ) {
            (gas_price, max_fee_per_gas, max_priority_fee_per_gas, access_list)
                if data.evm_spec_id() >= l1::SpecId::LONDON
                    && (gas_price.is_none()
                        || max_fee_per_gas.is_some()
                        || max_priority_fee_per_gas.is_some()) =>
            {
                let (max_fee_per_gas, max_priority_fee_per_gas) =
                    match (max_fee_per_gas, max_priority_fee_per_gas) {
                        (Some(max_fee_per_gas), Some(max_priority_fee_per_gas)) => {
                            (max_fee_per_gas, max_priority_fee_per_gas)
                        }
                        (Some(max_fee_per_gas), None) => (
                            max_fee_per_gas,
                            max_fee_per_gas.min(U256::from(DEFAULT_MAX_PRIORITY_FEE_PER_GAS)),
                        ),
                        (None, Some(max_priority_fee_per_gas)) => {
                            let max_fee_per_gas =
                                calculate_max_fee_per_gas(data, max_priority_fee_per_gas)?;
                            (max_fee_per_gas, max_priority_fee_per_gas)
                        }
                        (None, None) => {
                            let max_priority_fee_per_gas =
                                U256::from(DEFAULT_MAX_PRIORITY_FEE_PER_GAS);
                            let max_fee_per_gas =
                                calculate_max_fee_per_gas(data, max_priority_fee_per_gas)?;
                            (max_fee_per_gas, max_priority_fee_per_gas)
                        }
                    };

                transaction::Request::Eip1559(transaction::request::Eip1559 {
                    nonce,
                    max_priority_fee_per_gas,
                    max_fee_per_gas,
                    gas_limit,
                    value,
                    input,
                    kind: to.into(),
                    chain_id,
                    access_list: access_list.unwrap_or_default(),
                })
            }
            (gas_price, _, _, Some(access_list)) => {
                transaction::Request::Eip2930(transaction::request::Eip2930 {
                    nonce,
                    gas_price: gas_price.map_or_else(|| data.next_gas_price(), Ok)?,
                    gas_limit,
                    value,
                    input,
                    kind: to.into(),
                    chain_id,
                    access_list,
                })
            }
            (gas_price, _, _, _) => transaction::Request::Eip155(transaction::request::Eip155 {
                nonce,
                gas_price: gas_price.map_or_else(|| data.next_gas_price(), Ok)?,
                gas_limit,
                value,
                input,
                kind: to.into(),
                chain_id,
            }),
        };

        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use edr_eth::BlockSpec;
    use edr_evm::state::StateOverrides;
    use edr_rpc_eth::CallRequest;

    use super::*;
    use crate::test_utils::{pending_base_fee, ProviderTestFixture};

    #[test]
    fn resolve_call_request_with_gas_price() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::new_local()?;

        let pending_base_fee = pending_base_fee(&mut fixture.provider_data)?;

        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            gas_price: Some(pending_base_fee),
            ..CallRequest::default()
        };

        let context = CallContext {
            data: &mut fixture.provider_data,
            block_spec: &BlockSpec::pending(),
            state_overrides: &StateOverrides::default(),
            default_gas_price_fn: |_data| unreachable!("gas_price is set"),
            max_fees_fn: |_, _, _, _| unreachable!("gas_price is set"),
        };

        let resolved = transaction::Request::from_rpc_type(request, context)?;
        assert_eq!(*resolved.gas_price(), pending_base_fee);

        Ok(())
    }

    #[test]
    fn resolve_call_request_inner_with_max_fee_and_max_priority_fee() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::new_local()?;

        let max_fee_per_gas = pending_base_fee(&mut fixture.provider_data)?;
        let max_priority_fee_per_gas = Some(max_fee_per_gas / U256::from(2));

        let request = CallRequest {
            from: Some(fixture.nth_local_account(0)?),
            to: Some(fixture.nth_local_account(1)?),
            max_fee_per_gas: Some(max_fee_per_gas),
            max_priority_fee_per_gas,
            ..CallRequest::default()
        };

        let context = CallContext {
            data: &mut fixture.provider_data,
            block_spec: &BlockSpec::pending(),
            state_overrides: &StateOverrides::default(),
            default_gas_price_fn: |_data| unreachable!("max fees are set"),
            max_fees_fn: |_data, _block_spec, max_fee_per_gas, max_priority_fee_per_gas| {
                Ok((
                    max_fee_per_gas.expect("max fee is set"),
                    max_priority_fee_per_gas.expect("max priority fee is set"),
                ))
            },
        };

        let resolved = transaction::Request::from_rpc_type(request, context)?;

        assert_eq!(*resolved.gas_price(), max_fee_per_gas);
        assert_eq!(
            resolved.max_priority_fee_per_gas().cloned(),
            max_priority_fee_per_gas
        );

        Ok(())
    }
}
