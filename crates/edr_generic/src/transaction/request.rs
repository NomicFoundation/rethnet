use edr_eth::{
    l1,
    signature::{SecretKey, SignatureError},
    transaction::signed::{FakeSign, Sign},
    Address, Bytes, U256,
};
use edr_evm::blockchain::BlockchainErrorForChainSpec;
use edr_provider::{
    requests::validation::{validate_call_request, validate_send_transaction_request},
    spec::{CallContext, FromRpcType, TransactionContext},
    time::TimeSinceEpoch,
    ProviderData, ProviderError,
};
use edr_rpc_eth::{CallRequest, TransactionRequest};

use crate::{transaction::SignedWithFallbackToPostEip155, GenericChainSpec};

/// Container type for various Ethereum transaction requests.
// NOTE: This is a newtype only because the default FromRpcType implementation
// provides an error of ProviderError<L1ChainSpec> specifically. Despite us
// wanting the same logic, we need to use our own type and copy the
// implementation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Request(edr_eth::transaction::Request);

impl From<edr_eth::transaction::Request> for Request {
    fn from(value: edr_eth::transaction::Request) -> Self {
        Self(value)
    }
}

impl FakeSign for Request {
    type Signed = SignedWithFallbackToPostEip155;

    fn fake_sign(self, sender: edr_eth::Address) -> SignedWithFallbackToPostEip155 {
        <edr_eth::transaction::Request as FakeSign>::fake_sign(self.0, sender).into()
    }
}

impl Sign for Request {
    type Signed = SignedWithFallbackToPostEip155;

    unsafe fn sign_for_sender_unchecked(
        self,
        secret_key: &SecretKey,
        caller: Address,
    ) -> Result<SignedWithFallbackToPostEip155, SignatureError> {
        <edr_eth::transaction::Request as Sign>::sign_for_sender_unchecked(
            self.0, secret_key, caller,
        )
        .map(Into::into)
    }
}

impl<TimerT: Clone + TimeSinceEpoch> FromRpcType<CallRequest, TimerT> for Request {
    type Context<'context> = CallContext<'context, GenericChainSpec, TimerT>;

    type Error = ProviderError<GenericChainSpec>;

    fn from_rpc_type(
        value: CallRequest,
        context: Self::Context<'_>,
    ) -> Result<crate::transaction::Request, ProviderError<GenericChainSpec>> {
        let CallContext {
            data,
            block_spec,
            state_overrides,
            default_gas_price_fn,
            max_fees_fn,
        } = context;

        validate_call_request(data.hardfork(), &value, block_spec)?;

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
                    edr_eth::transaction::Request::Eip2930(edr_eth::transaction::request::Eip2930 {
                        nonce,
                        gas_price,
                        gas_limit,
                        value,
                        input,
                        kind: to.into(),
                        chain_id,
                        access_list,
                    })
                    .into()
                }
                _ => edr_eth::transaction::Request::Eip155(edr_eth::transaction::request::Eip155 {
                    nonce,
                    gas_price,
                    gas_limit,
                    kind: to.into(),
                    value,
                    input,
                    chain_id,
                })
                .into(),
            }
        } else {
            let (max_fee_per_gas, max_priority_fee_per_gas) =
                max_fees_fn(data, block_spec, max_fee_per_gas, max_priority_fee_per_gas)?;

            edr_eth::transaction::Request::Eip1559(edr_eth::transaction::request::Eip1559 {
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
            .into()
        };

        Ok(request)
    }
}

impl<TimerT: Clone + TimeSinceEpoch> FromRpcType<TransactionRequest, TimerT> for Request {
    type Context<'context> = TransactionContext<'context, GenericChainSpec, TimerT>;

    type Error = ProviderError<GenericChainSpec>;

    fn from_rpc_type(
        value: TransactionRequest,
        context: Self::Context<'_>,
    ) -> Result<crate::transaction::Request, ProviderError<GenericChainSpec>> {
        const DEFAULT_MAX_PRIORITY_FEE_PER_GAS: u64 = 1_000_000_000;

        /// # Panics
        ///
        /// Panics if `data.evm_spec_id()` is less than `SpecId::LONDON`.
        fn calculate_max_fee_per_gas<TimerT: Clone + TimeSinceEpoch>(
            data: &ProviderData<GenericChainSpec, TimerT>,
            max_priority_fee_per_gas: U256,
        ) -> Result<U256, BlockchainErrorForChainSpec<GenericChainSpec>> {
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

                edr_eth::transaction::Request::Eip1559(edr_eth::transaction::request::Eip1559 {
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
                .into()
            }
            (gas_price, _, _, Some(access_list)) => {
                edr_eth::transaction::Request::Eip2930(edr_eth::transaction::request::Eip2930 {
                    nonce,
                    gas_price: gas_price.map_or_else(|| data.next_gas_price(), Ok)?,
                    gas_limit,
                    value,
                    input,
                    kind: to.into(),
                    chain_id,
                    access_list,
                })
                .into()
            }
            (gas_price, _, _, _) => {
                edr_eth::transaction::Request::Eip155(edr_eth::transaction::request::Eip155 {
                    nonce,
                    gas_price: gas_price.map_or_else(|| data.next_gas_price(), Ok)?,
                    gas_limit,
                    value,
                    input,
                    kind: to.into(),
                    chain_id,
                })
                .into()
            }
        };

        Ok(request)
    }
}
