use std::sync::Arc;

use edr_eth::{
    result::InvalidTransaction,
    rlp::Decodable,
    transaction::{
        request::TransactionRequestAndSender, IsEip155, IsEip4844, Transaction as _,
        TransactionType, TransactionValidation, INVALID_TX_TYPE_ERROR_MESSAGE,
    },
    Bytes, PreEip1898BlockSpec, B256, U256,
};
use edr_evm::{
    block::transaction::{BlockDataForTransaction, TransactionAndBlock},
    spec::RuntimeSpec,
    transaction, Block,
};
use edr_rpc_eth::RpcTypeFrom as _;

use crate::{
    data::ProviderData,
    error::TransactionFailureWithTraces,
    requests::validation::{
        validate_eip3860_max_initcode_size, validate_post_merge_block_tags,
        validate_transaction_and_call_request,
    },
    spec::{FromRpcType, Sender as _, SyncProviderSpec, TransactionContext},
    time::TimeSinceEpoch,
    ProviderError, ProviderResultWithTraces, TransactionFailure,
};

pub fn handle_get_transaction_by_block_hash_and_index<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    block_hash: B256,
    index: U256,
) -> Result<Option<ChainSpecT::RpcTransaction>, ProviderError<ChainSpecT>> {
    let index = rpc_index_to_usize(&index)?;

    let transaction = data
        .block_by_hash(&block_hash)?
        .and_then(|block| transaction_from_block(block, index, false))
        .map(|transaction_and_block| {
            ChainSpecT::RpcTransaction::rpc_type_from(&transaction_and_block, data.hardfork())
        });

    Ok(transaction)
}

pub fn handle_get_transaction_by_block_spec_and_index<
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
    block_spec: PreEip1898BlockSpec,
    index: U256,
) -> Result<Option<ChainSpecT::RpcTransaction>, ProviderError<ChainSpecT>> {
    validate_post_merge_block_tags(data.hardfork(), &block_spec)?;

    let index = rpc_index_to_usize(&index)?;

    let transaction = match data.block_by_block_spec(&block_spec.into()) {
        Ok(Some(block)) => Some((block, false)),
        // Pending block requested
        Ok(None) => {
            let result = data.mine_pending_block()?;
            let pending_block = Arc::new(result.block);
            Some((ChainSpecT::cast_local_block(pending_block), true))
        }
        // Matching Hardhat behavior in returning None for invalid block hash or number.
        Err(ProviderError::InvalidBlockNumberOrHash { .. }) => None,
        Err(err) => return Err(err),
    }
    .and_then(|(block, is_pending)| transaction_from_block(block, index, is_pending))
    .map(|transaction_and_block| {
        ChainSpecT::RpcTransaction::rpc_type_from(&transaction_and_block, data.hardfork())
    });

    Ok(transaction)
}

pub fn handle_pending_transactions<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
) -> Result<Vec<ChainSpecT::RpcTransaction>, ProviderError<ChainSpecT>> {
    let transactions = data
        .pending_transactions()
        .map(|pending_transaction| {
            let transaction_and_block = TransactionAndBlock {
                transaction: pending_transaction.clone(),
                block_data: None,
                is_pending: true,
            };
            ChainSpecT::RpcTransaction::rpc_type_from(&transaction_and_block, data.hardfork())
        })
        .collect();

    Ok(transactions)
}

fn rpc_index_to_usize<ChainSpecT: RuntimeSpec>(
    index: &U256,
) -> Result<usize, ProviderError<ChainSpecT>> {
    index
        .try_into()
        .map_err(|_err| ProviderError::InvalidTransactionIndex(*index))
}

pub fn handle_get_transaction_by_hash<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    transaction_hash: B256,
) -> Result<Option<ChainSpecT::RpcTransaction>, ProviderError<ChainSpecT>> {
    let transaction = data
        .transaction_by_hash(&transaction_hash)?
        .map(|transaction_and_block| {
            ChainSpecT::RpcTransaction::rpc_type_from(&transaction_and_block, data.hardfork())
        });

    Ok(transaction)
}

pub fn handle_get_transaction_receipt<
    ChainSpecT: SyncProviderSpec<TimerT>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    transaction_hash: B256,
) -> Result<Option<ChainSpecT::RpcReceipt>, ProviderError<ChainSpecT>> {
    let receipt = data.transaction_receipt(&transaction_hash)?;

    let rpc_receipt =
        receipt.map(|receipt| ChainSpecT::RpcReceipt::rpc_type_from(&receipt, data.hardfork()));

    Ok(rpc_receipt)
}

fn transaction_from_block<BlockT: Block<SignedTransactionT> + Clone, SignedTransactionT: Clone>(
    block: BlockT,
    transaction_index: usize,
    is_pending: bool,
) -> Option<TransactionAndBlock<BlockT, SignedTransactionT>> {
    block
        .transactions()
        .get(transaction_index)
        .map(|transaction| TransactionAndBlock {
            transaction: transaction.clone(),
            block_data: Some(BlockDataForTransaction {
                block: block.clone(),
                transaction_index: transaction_index.try_into().expect("usize fits into u64"),
            }),
            is_pending,
        })
}

pub fn handle_send_transaction_request<
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionType<Type: IsEip4844>
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    request: ChainSpecT::RpcTransactionRequest,
) -> ProviderResultWithTraces<B256, ChainSpecT> {
    let sender = *request.sender();

    let context = TransactionContext { data };
    let request = ChainSpecT::TransactionRequest::from_rpc_type(request, context)?;

    let request = TransactionRequestAndSender { request, sender };
    let signed_transaction = data.sign_transaction_request(request)?;

    send_raw_transaction_and_log(data, signed_transaction)
}

pub fn handle_send_raw_transaction_request<
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionType<Type: IsEip4844>
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
        PooledTransaction: IsEip155,
    >,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    raw_transaction: Bytes,
) -> ProviderResultWithTraces<B256, ChainSpecT> {
    let mut raw_transaction: &[u8] = raw_transaction.as_ref();
    let pooled_transaction =
    ChainSpecT::PooledTransaction::decode(&mut raw_transaction).map_err(|err| match err {
            edr_eth::rlp::Error::Custom(INVALID_TX_TYPE_ERROR_MESSAGE) => {
                let type_id = *raw_transaction.first().expect("We already validated that the transaction is not empty if it's an invalid transaction type error.");
                ProviderError::InvalidTransactionType(type_id)
            }
            err => ProviderError::InvalidArgument(err.to_string()),
        })?;

    validate_send_raw_transaction_request(data, &pooled_transaction)?;
    let signed_transaction = pooled_transaction.into();

    let signed_transaction = transaction::validate(signed_transaction, data.evm_spec_id())
        .map_err(ProviderError::TransactionCreationError)?;

    send_raw_transaction_and_log(data, signed_transaction)
}

fn send_raw_transaction_and_log<
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionType<Type: IsEip4844>
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &mut ProviderData<ChainSpecT, TimerT>,
    signed_transaction: ChainSpecT::SignedTransaction,
) -> ProviderResultWithTraces<B256, ChainSpecT> {
    let result = data.send_transaction(signed_transaction.clone())?;

    let hardfork = data.hardfork();
    data.logger_mut()
        .log_send_transaction(hardfork, &signed_transaction, &result.mining_results)
        .map_err(ProviderError::Logger)?;

    if data.bail_on_transaction_failure() {
        let transaction_failure =
            result
                .transaction_result_and_trace()
                .and_then(|(execution_result, trace)| {
                    TransactionFailure::from_execution_result::<ChainSpecT, TimerT>(
                        execution_result,
                        Some(&result.transaction_hash),
                        trace,
                    )
                });

        if let Some(failure) = transaction_failure {
            let (_transaction_hash, traces) = result.into();
            return Err(ProviderError::TransactionFailed(
                TransactionFailureWithTraces { failure, traces },
            ));
        }
    }

    Ok(result.into())
}

fn validate_send_raw_transaction_request<
    ChainSpecT: SyncProviderSpec<TimerT, PooledTransaction: IsEip155>,
    TimerT: Clone + TimeSinceEpoch,
>(
    data: &ProviderData<ChainSpecT, TimerT>,
    transaction: &ChainSpecT::PooledTransaction,
) -> Result<(), ProviderError<ChainSpecT>> {
    if let Some(tx_chain_id) = transaction.chain_id() {
        let expected = data.chain_id();
        if tx_chain_id != expected {
            let error = if transaction.is_eip155() {
                ProviderError::InvalidEip155TransactionChainId
            } else {
                ProviderError::InvalidArgument(format!("Trying to send a raw transaction with an invalid chainId. The expected chainId is {expected}"))
            };
            return Err(error);
        }
    }

    validate_eip3860_max_initcode_size(
        data.evm_spec_id(),
        data.allow_unlimited_initcode_size(),
        transaction.kind().to(),
        transaction.data(),
    )?;

    validate_transaction_and_call_request(data.hardfork(), transaction).map_err(|err| match err {
        ProviderError::UnsupportedEIP1559Parameters {
            minimum_hardfork, ..
        } => ProviderError::InvalidArgument(format!(
            "\
Trying to send an EIP-1559 transaction but they are not supported by the current hard fork.\
\
You can use them by running Hardhat Network with 'hardfork' {minimum_hardfork:?} or later."
        )),
        err => err,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use edr_eth::{l1::L1ChainSpec, Address, Bytes, U256};
    use transaction::{signed::FakeSign as _, TxKind};

    use super::*;
    use crate::test_utils::{one_ether, ProviderTestFixture};

    #[test]
    fn transaction_by_hash_for_impersonated_account() -> anyhow::Result<()> {
        let mut fixture = ProviderTestFixture::<L1ChainSpec>::new_local()?;

        let impersonated_account: Address = "0x20620fa0ad46516e915029c94e3c87c9cd7861ff".parse()?;
        fixture
            .provider_data
            .impersonate_account(impersonated_account);

        fixture
            .provider_data
            .set_balance(impersonated_account, one_ether())?;

        let chain_id = fixture.provider_data.chain_id();

        let transaction = transaction::Request::Eip155(transaction::request::Eip155 {
            kind: TxKind::Call(Address::ZERO),
            gas_limit: 30_000,
            gas_price: U256::from(42_000_000_000_u64),
            value: U256::from(1),
            input: Bytes::default(),
            nonce: 0,
            chain_id,
        })
        .fake_sign(impersonated_account);
        let transaction = transaction::validate(transaction, fixture.provider_data.evm_spec_id())?;

        fixture.provider_data.set_auto_mining(true);
        let result = fixture.provider_data.send_transaction(transaction)?;
        assert!(result.transaction_result_and_trace().is_some());

        let rpc_transaction =
            handle_get_transaction_by_hash(&fixture.provider_data, result.transaction_hash)?
                .context("transaction not found")?;
        assert_eq!(&rpc_transaction.from, &impersonated_account);
        assert_eq!(&rpc_transaction.hash, &result.transaction_hash);

        Ok(())
    }
}
