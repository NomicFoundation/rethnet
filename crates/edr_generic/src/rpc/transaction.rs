use edr_eth::{l1, transaction::SignedTransaction as _};
use edr_evm::{
    block::transaction::{BlockDataForTransaction, TransactionAndBlockForChainSpec},
    transaction::remote::EthRpcTransaction,
};
use edr_rpc_eth::RpcTypeFrom;
use serde::{Deserialize, Serialize};

use crate::{transaction, GenericChainSpec};

// We need to use a newtype here as `RpcTypeFrom` cannot be implemented here,
// in an external crate, even though `TransactionAndBlock` is generic over
// a type that we introduced.
// This originally works as the impl for `L1ChainSpec` lives already in the
// defining crate of `edr_evm::TransactionAndBlock`, which probably shouldn't
// as far as defining spec externally is concerned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TransactionWithSignature(edr_rpc_eth::TransactionWithSignature);

impl EthRpcTransaction for TransactionWithSignature {
    fn block_hash(&self) -> Option<&edr_eth::B256> {
        self.0.block_hash()
    }
}

impl From<edr_rpc_eth::TransactionWithSignature> for TransactionWithSignature {
    fn from(value: edr_rpc_eth::TransactionWithSignature) -> Self {
        Self(value)
    }
}

impl RpcTypeFrom<TransactionAndBlockForChainSpec<GenericChainSpec>> for TransactionWithSignature {
    type Hardfork = l1::SpecId;

    fn rpc_type_from(
        value: &TransactionAndBlockForChainSpec<GenericChainSpec>,
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

        let transaction = edr_rpc_eth::Transaction::new(
            &value.transaction,
            header,
            transaction_index,
            value.is_pending,
            hardfork,
        );
        let signature = value.transaction.signature();

        edr_rpc_eth::TransactionWithSignature::new(
            transaction,
            signature.r(),
            signature.s(),
            signature.v(),
            signature.y_parity(),
        )
        .into()
    }
}

pub use edr_rpc_eth::TransactionConversionError as ConversionError;

impl TryFrom<TransactionWithSignature> for transaction::SignedWithFallbackToPostEip155 {
    type Error = ConversionError;

    fn try_from(value: TransactionWithSignature) -> Result<Self, Self::Error> {
        use edr_eth::transaction::Signed;

        let TransactionWithSignature(value) = value;

        let tx_type = match value
            .transaction_type
            .map(edr_eth::transaction::Type::try_from)
        {
            None => transaction::Type::Legacy,
            Some(Ok(r#type)) => r#type.into(),
            Some(Err(r#type)) => {
                log::warn!("Unsupported transaction type: {type}. Reverting to post-EIP 155 legacy transaction");

                transaction::Type::Unrecognized(r#type)
            }
        };

        let transaction = match tx_type {
            // We explicitly treat unrecognized transaction types as post-EIP 155 legacy
            // transactions
            transaction::Type::Unrecognized(_) => Signed::PostEip155Legacy(value.into()),

            transaction::Type::Legacy => {
                if value.is_legacy() {
                    Signed::PreEip155Legacy(value.into())
                } else {
                    Signed::PostEip155Legacy(value.into())
                }
            }
            transaction::Type::Eip2930 => Signed::Eip2930(value.try_into()?),
            transaction::Type::Eip1559 => Signed::Eip1559(value.try_into()?),
            transaction::Type::Eip4844 => Signed::Eip4844(value.try_into()?),
        };

        Ok(Self::with_type(transaction, tx_type))
    }
}
