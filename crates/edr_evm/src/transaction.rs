mod detailed;
/// Types for transactions from a remote provider.
pub mod remote;

use std::fmt::Debug;

use derive_where::derive_where;
// Re-export the transaction types from `edr_eth`.
pub use edr_eth::transaction::*;
use edr_eth::{l1, spec::ChainSpec, U256};
use revm::interpreter::gas::validate_initial_tx_gas;

pub use self::detailed::*;
use crate::{
    result::{EVMError, EVMErrorForChain, InvalidHeader},
    spec::RuntimeSpec,
    state::DatabaseComponentError,
};

/// Invalid transaction error
#[derive(thiserror::Error)]
#[derive_where(Debug; <ChainSpecT::SignedTransaction as TransactionValidation>::ValidationError, BlockchainErrorT, StateErrorT)]
pub enum TransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: ChainSpec,
{
    /// Blockchain errors
    #[error(transparent)]
    Blockchain(BlockchainErrorT),
    /// Custom errors
    #[error("{0}")]
    Custom(String),
    /// EIP-1559 is not supported
    #[error("Cannot run transaction: EIP 1559 is not activated.")]
    Eip1559Unsupported,
    /// Invalid block header
    #[error(transparent)]
    InvalidHeader(InvalidHeader),
    /// Corrupt transaction data
    #[error(transparent)]
    InvalidTransaction(<ChainSpecT::SignedTransaction as TransactionValidation>::ValidationError),
    /// Transaction account does not have enough amount of ether to cover
    /// transferred value and gas_limit*gas_price.
    #[error("Sender doesn't have enough funds to send tx. The max upfront cost is: {fee} and the sender's balance is: {balance}.")]
    LackOfFundForMaxFee {
        /// The max upfront cost of the transaction
        fee: Box<U256>,
        /// The sender's balance
        balance: Box<U256>,
    },
    /// Precompile errors
    #[error("{0}")]
    Precompile(String),
    /// State errors
    #[error(transparent)]
    State(StateErrorT),
}

impl<ChainSpecT, BlockchainErrorT, StateErrorT>
    From<EVMErrorForChain<ChainSpecT, BlockchainErrorT, StateErrorT>>
    for TransactionError<ChainSpecT, BlockchainErrorT, StateErrorT>
where
    ChainSpecT: RuntimeSpec,
{
    fn from(error: EVMErrorForChain<ChainSpecT, BlockchainErrorT, StateErrorT>) -> Self {
        match error {
            EVMError::Transaction(error) => ChainSpecT::cast_transaction_error(error),
            EVMError::Header(error) => Self::InvalidHeader(error),
            EVMError::Database(DatabaseComponentError::Blockchain(e)) => Self::Blockchain(e),
            EVMError::Database(DatabaseComponentError::State(e)) => Self::State(e),
            EVMError::Custom(error) => Self::Custom(error),
            EVMError::Precompile(error) => Self::Precompile(error),
        }
    }
}

/// An error that occurred while during [`validate`].
#[derive(Debug, thiserror::Error)]
pub enum CreationError {
    /// Creating contract without any data.
    #[error("Contract creation without any data provided")]
    ContractMissingData,
    /// Transaction gas limit is insufficient to afford initial gas cost.
    #[error("Transaction requires at least {initial_gas_cost} gas but got {gas_limit}")]
    InsufficientGas {
        /// The initial gas cost of a transaction
        initial_gas_cost: U256,
        /// The gas limit of the transaction
        gas_limit: u64,
    },
}

/// Validates the transaction.
pub fn validate<TransactionT: Transaction>(
    transaction: TransactionT,
    spec_id: l1::SpecId,
) -> Result<TransactionT, CreationError> {
    if transaction.kind() == TxKind::Create && transaction.data().is_empty() {
        return Err(CreationError::ContractMissingData);
    }

    let initial_cost = initial_cost(&transaction, spec_id);
    if transaction.gas_limit() < initial_cost {
        return Err(CreationError::InsufficientGas {
            initial_gas_cost: U256::from(initial_cost),
            gas_limit: transaction.gas_limit(),
        });
    }

    Ok(transaction)
}

/// Calculates the initial cost of a transaction.
pub fn initial_cost(transaction: &impl Transaction, spec_id: l1::SpecId) -> u64 {
    validate_initial_tx_gas(
        spec_id,
        transaction.data().as_ref(),
        transaction.kind() == TxKind::Create,
        transaction.access_list(),
        0,
    )
}

#[cfg(test)]
mod tests {
    use edr_eth::{transaction, Address, Bytes};

    use super::*;

    #[test]
    fn gas_limit_less_than_base_fee() -> anyhow::Result<()> {
        const TOO_LOW_GAS_LIMIT: u64 = 100;

        let caller = Address::random();

        let request = transaction::request::Eip155 {
            nonce: 0,
            gas_price: U256::ZERO,
            gas_limit: TOO_LOW_GAS_LIMIT,
            kind: TxKind::Call(caller),
            value: U256::ZERO,
            input: Bytes::new(),
            chain_id: 123,
        };

        let transaction = request.fake_sign(caller);
        let transaction = transaction::Signed::from(transaction);
        let result = validate(transaction, l1::SpecId::BERLIN);

        let expected_gas_cost = U256::from(21_000);
        assert!(matches!(
            result,
            Err(CreationError::InsufficientGas {
                initial_gas_cost,
                gas_limit: TOO_LOW_GAS_LIMIT,
            }) if initial_gas_cost == expected_gas_cost
        ));

        assert_eq!(
            result.unwrap_err().to_string(),
            format!("Transaction requires at least 21000 gas but got {TOO_LOW_GAS_LIMIT}")
        );

        Ok(())
    }

    #[test]
    fn create_missing_data() -> anyhow::Result<()> {
        let caller = Address::random();

        let request = transaction::request::Eip155 {
            nonce: 0,
            gas_price: U256::ZERO,
            gas_limit: 30_000,
            kind: TxKind::Create,
            value: U256::ZERO,
            input: Bytes::new(),
            chain_id: 123,
        };

        let transaction = request.fake_sign(caller);
        let transaction = transaction::Signed::from(transaction);
        let result = validate(transaction, l1::SpecId::BERLIN);

        assert!(matches!(result, Err(CreationError::ContractMissingData)));

        assert_eq!(
            result.unwrap_err().to_string(),
            "Contract creation without any data provided"
        );

        Ok(())
    }
}
