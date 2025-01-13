use std::sync::OnceLock;

use edr_eth::{signature::Signature, transaction::MaybeSignedTransaction, B256};
use edr_evm::{
    block::transaction::{BlockDataForTransaction, TransactionAndBlockForChainSpec},
    transaction::{remote::EthRpcTransaction, TxKind},
};
use edr_rpc_eth::{
    RpcTypeFrom, TransactionConversionError as L1ConversionError, TransactionWithSignature,
};
use revm_optimism::{OptimismSpecId, OptimismTransaction};

use super::Transaction;
use crate::{transaction, OptimismChainSpec};

impl EthRpcTransaction for Transaction {
    fn block_hash(&self) -> Option<&B256> {
        self.l1.block_hash.as_ref()
    }
}

impl TryFrom<Transaction> for transaction::signed::Deposit {
    type Error = ConversionError;

    fn try_from(value: Transaction) -> Result<Self, Self::Error> {
        let transaction = Self {
            source_hash: value.source_hash.ok_or(ConversionError::SourceHash)?,
            from: value.l1.from,
            to: if let Some(to) = value.l1.to {
                TxKind::Call(to)
            } else {
                TxKind::Create
            },
            mint: value.mint.unwrap_or_default(),
            value: value.l1.value,
            gas_limit: value.l1.gas.to(),
            is_system_tx: value.is_system_tx.unwrap_or(false),
            data: value.l1.input,
            hash: OnceLock::new(),
            rlp_encoding: OnceLock::new(),
        };

        Ok(transaction)
    }
}

/// Error that occurs when trying to convert the JSON-RPC `Transaction` type.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// L1 conversion error
    #[error(transparent)]
    L1(#[from] L1ConversionError),
    /// Missing mint
    #[error("Missing mint")]
    Mint,
    /// Missing signature R-value.
    #[error("Missing signature R-value")]
    SignatureR,
    /// Missing signature S-value.
    #[error("Missing signature S-value")]
    SignatureS,
    /// Missing signature V-value.
    #[error("Missing signature V-value")]
    SignatureV,
    /// Missing source hash
    #[error("Missing source hash")]
    SourceHash,
}

impl TryFrom<Transaction> for transaction::Signed {
    type Error = ConversionError;

    fn try_from(value: Transaction) -> Result<Self, Self::Error> {
        let transaction_type = match value
            .l1
            .transaction_type
            .map_or(Ok(transaction::Type::Legacy), transaction::Type::try_from)
        {
            Ok(r#type) => r#type,
            Err(r#type) => {
                log::warn!("Unsupported transaction type: {type}. Reverting to post-EIP 155 legacy transaction");

                // As the transaction type is not 0 or `None`, this will always result in a
                // post-EIP 155 legacy transaction.
                transaction::Type::Legacy
            }
        };

        let transaction = match transaction_type {
            transaction::Type::Deposit => Self::Deposit(value.try_into()?),
            transaction_type => {
                let r = value.r.ok_or(ConversionError::SignatureR)?;
                let s = value.s.ok_or(ConversionError::SignatureS)?;
                let v = value.v.ok_or(ConversionError::SignatureV)?;

                let transaction_with_signature =
                    TransactionWithSignature::new(value.l1, r, s, v, value.y_parity);

                match transaction_type {
                    transaction::Type::Legacy => {
                        if transaction_with_signature.is_legacy() {
                            Self::PreEip155Legacy(transaction_with_signature.into())
                        } else {
                            Self::PostEip155Legacy(transaction_with_signature.into())
                        }
                    }
                    transaction::Type::Eip2930 => {
                        Self::Eip2930(transaction_with_signature.try_into()?)
                    }
                    transaction::Type::Eip1559 => {
                        Self::Eip1559(transaction_with_signature.try_into()?)
                    }
                    transaction::Type::Eip4844 => {
                        Self::Eip4844(transaction_with_signature.try_into()?)
                    }
                    transaction::Type::Deposit => unreachable!("already handled"),
                }
            }
        };

        Ok(transaction)
    }
}

impl RpcTypeFrom<TransactionAndBlockForChainSpec<OptimismChainSpec>> for Transaction {
    type Hardfork = OptimismSpecId;

    fn rpc_type_from(
        value: &TransactionAndBlockForChainSpec<OptimismChainSpec>,
        hardfork: Self::Hardfork,
    ) -> Self {
        let (header, transaction_index) = value
            .block_data
            .as_ref()
            .map(
                |BlockDataForTransaction {
                     block,
                     transaction_index,
                 }| (block.header(), *transaction_index),
            )
            .unzip();

        let l1 = edr_rpc_eth::Transaction::new(
            &value.transaction,
            header,
            transaction_index,
            value.is_pending,
            hardfork.into(),
        );

        let signature = value.transaction.maybe_signature();

        Self {
            l1,
            v: signature.map(Signature::v),
            // Following Hardhat in always returning `v` instead of `y_parity`.
            y_parity: None,
            r: signature.map(Signature::r),
            s: signature.map(Signature::s),
            source_hash: value.transaction.source_hash().copied(),
            mint: value.transaction.mint().copied(),
            is_system_tx: value.transaction.is_system_transaction(),
        }
    }
}
