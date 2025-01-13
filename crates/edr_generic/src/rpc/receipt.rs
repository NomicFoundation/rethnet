use edr_eth::{l1, log::FilterLog, receipt::AsExecutionReceipt as _};
use edr_rpc_eth::RpcTypeFrom;
use serde::{Deserialize, Serialize};

use crate::eip2718::TypedEnvelope;

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Legacy transaction is missing state root or status")]
    MissingStateRootOrStatus,
    #[error("Missing status")]
    MissingStatus,
}

use edr_eth::{
    receipt::{
        execution::{Eip658, Legacy},
        Execution, ExecutionReceipt, TransactionReceipt,
    },
    transaction::TransactionType,
};

// We need to introduce a newtype for BlockReceipt again due to the orphan rule,
// even though we use our own TypedEnvelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockReceipt(edr_rpc_eth::receipt::Block);

impl TryFrom<BlockReceipt> for crate::receipt::BlockReceipt<TypedEnvelope<Execution<FilterLog>>> {
    type Error = ConversionError;

    fn try_from(value: BlockReceipt) -> Result<Self, Self::Error> {
        let BlockReceipt(value) = value;

        // We explicitly treat unknown transaction types as post-EIP 155 legacy
        // transactions
        let transaction_type = value.transaction_type.map_or(
            crate::transaction::Type::Legacy,
            crate::transaction::Type::from,
        );

        let execution = if transaction_type == crate::transaction::Type::Legacy {
            if let Some(status) = value.status {
                Execution::Eip658(Eip658 {
                    status,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                })
            } else if let Some(state_root) = value.state_root {
                Execution::Legacy(Legacy {
                    root: state_root,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                })
            } else {
                return Err(ConversionError::MissingStateRootOrStatus);
            }
        } else {
            Execution::Eip658(Eip658 {
                status: value.status.ok_or(ConversionError::MissingStatus)?,
                cumulative_gas_used: value.cumulative_gas_used,
                logs_bloom: value.logs_bloom,
                logs: value.logs,
            })
        };

        let enveloped = TypedEnvelope::new(execution, transaction_type);

        Ok(Self {
            block_hash: value.block_hash,
            block_number: value.block_number,
            inner: TransactionReceipt {
                inner: enveloped,
                transaction_hash: value.transaction_hash,
                transaction_index: value.transaction_index,
                from: value.from,
                to: value.to,
                contract_address: value.contract_address,
                gas_used: value.gas_used,
                effective_gas_price: value.effective_gas_price,
            },
        })
    }
}

impl RpcTypeFrom<crate::receipt::BlockReceipt<TypedEnvelope<Execution<FilterLog>>>>
    for BlockReceipt
{
    type Hardfork = l1::SpecId;

    fn rpc_type_from(
        value: &crate::receipt::BlockReceipt<TypedEnvelope<Execution<FilterLog>>>,
        hardfork: Self::Hardfork,
    ) -> Self {
        let transaction_type = if hardfork >= l1::SpecId::BERLIN {
            Some(u8::from(value.inner.transaction_type()))
        } else {
            None
        };

        BlockReceipt(edr_rpc_eth::receipt::Block {
            block_hash: value.block_hash,
            block_number: value.block_number,
            transaction_hash: value.inner.transaction_hash,
            transaction_index: value.inner.transaction_index,
            transaction_type,
            from: value.inner.from,
            to: value.inner.to,
            cumulative_gas_used: value.inner.cumulative_gas_used(),
            gas_used: value.inner.gas_used,
            contract_address: value.inner.contract_address,
            logs: value.inner.transaction_logs().to_vec(),
            logs_bloom: *value.inner.logs_bloom(),
            state_root: match value.as_execution_receipt().data() {
                Execution::Legacy(receipt) => Some(receipt.root),
                Execution::Eip658(_) => None,
            },
            status: match value.as_execution_receipt().data() {
                Execution::Legacy(_) => None,
                Execution::Eip658(receipt) => Some(receipt.status),
            },
            effective_gas_price: value.inner.effective_gas_price,
            authorization_list: None,
        })
    }
}
