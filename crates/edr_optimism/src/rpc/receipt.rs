use edr_eth::{
    receipt::{AsExecutionReceipt as _, ExecutionReceipt as _, TransactionReceipt},
    transaction::TransactionType as _,
};
use edr_rpc_eth::RpcTypeFrom;
use revm_optimism::OptimismSpecId;

use crate::{eip2718::TypedEnvelope, receipt, rpc, transaction};

impl RpcTypeFrom<receipt::Block> for rpc::BlockReceipt {
    type Hardfork = OptimismSpecId;

    fn rpc_type_from(value: &receipt::Block, hardfork: Self::Hardfork) -> Self {
        let transaction_type = if hardfork >= OptimismSpecId::BERLIN {
            Some(u8::from(value.eth.inner.transaction_type()))
        } else {
            None
        };

        Self {
            block_hash: value.eth.block_hash,
            block_number: value.eth.block_number,
            transaction_hash: value.eth.inner.transaction_hash,
            transaction_index: value.eth.inner.transaction_index,
            transaction_type,
            from: value.eth.inner.from,
            to: value.eth.inner.to,
            cumulative_gas_used: value.eth.inner.cumulative_gas_used(),
            gas_used: value.eth.inner.gas_used,
            contract_address: value.eth.inner.contract_address,
            logs: value.eth.inner.transaction_logs().to_vec(),
            logs_bloom: *value.eth.inner.logs_bloom(),
            state_root: match value.as_execution_receipt().data() {
                receipt::Execution::Legacy(receipt) => Some(receipt.root),
                receipt::Execution::Eip658(_) | receipt::Execution::Deposit(_) => None,
            },
            status: match value.as_execution_receipt().data() {
                receipt::Execution::Legacy(_) => None,
                receipt::Execution::Eip658(receipt) => Some(receipt.status),
                receipt::Execution::Deposit(receipt) => Some(receipt.status),
            },
            effective_gas_price: value.eth.inner.effective_gas_price,
            deposit_nonce: match value.as_execution_receipt().data() {
                receipt::Execution::Legacy(_) | receipt::Execution::Eip658(_) => None,
                receipt::Execution::Deposit(receipt) => Some(receipt.deposit_nonce),
            },
            deposit_receipt_version: match value.as_execution_receipt().data() {
                receipt::Execution::Legacy(_) | receipt::Execution::Eip658(_) => None,
                receipt::Execution::Deposit(receipt) => receipt.deposit_receipt_version,
            },
            l1_block_info: value.l1_block_info.unwrap_or_default(),
            authorization_list: None,
        }
    }
}

/// Error type for conversions from an RPC receipt to a typed receipt.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Missing deposit nonce.
    ///
    /// Only occurs for deposit receipts.
    #[error("Missing deposit nonce")]
    MissingDepositNonce,
    /// Missing L1 block info.
    ///
    /// Only occurs for non-deposit receipts.
    #[error("Missing L1 block info for a non-deposit receipt")]
    MissingL1BlockInfo,
    /// Missing state root or status.
    ///
    /// Only occurs for legacy receipts.
    #[error("Legacy transaction is missing state root or status")]
    MissingStateRootOrStatus,
    /// Missing status.
    ///
    /// Only occurs for post-EIP-658 receipts.
    #[error("Missing status")]
    MissingStatus,
    /// Unknown transaction type.
    #[error("Unknown transaction type: {0}")]
    UnknownType(u8),
}

impl TryFrom<rpc::BlockReceipt> for receipt::Block {
    type Error = ConversionError;

    fn try_from(value: rpc::BlockReceipt) -> Result<Self, Self::Error> {
        let transaction_type = value
            .transaction_type
            .map_or(Ok(transaction::Type::Legacy), transaction::Type::try_from)
            .map_err(ConversionError::UnknownType)?;

        let (execution, l1_block_info) = match transaction_type {
            transaction::Type::Legacy => {
                let execution = if let Some(status) = value.status {
                    receipt::Execution::Eip658(receipt::execution::Eip658 {
                        status,
                        cumulative_gas_used: value.cumulative_gas_used,
                        logs_bloom: value.logs_bloom,
                        logs: value.logs,
                    })
                } else if let Some(state_root) = value.state_root {
                    receipt::Execution::Legacy(receipt::execution::Legacy {
                        root: state_root,
                        cumulative_gas_used: value.cumulative_gas_used,
                        logs_bloom: value.logs_bloom,
                        logs: value.logs,
                    })
                } else {
                    return Err(ConversionError::MissingStateRootOrStatus);
                };

                (execution, None)
            }
            transaction::Type::Deposit => {
                let execution = receipt::Execution::Deposit(receipt::execution::Deposit {
                    status: value.status.ok_or(ConversionError::MissingStatus)?,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                    deposit_nonce: value
                        .deposit_nonce
                        .ok_or(ConversionError::MissingDepositNonce)?,
                    deposit_receipt_version: value.deposit_receipt_version,
                });

                (execution, None)
            }
            _ => {
                let execution = receipt::Execution::Eip658(receipt::execution::Eip658 {
                    status: value.status.ok_or(ConversionError::MissingStatus)?,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                });

                (execution, Some(value.l1_block_info))
            }
        };

        let enveloped = TypedEnvelope::new(execution, transaction_type);

        let eth = edr_eth::receipt::BlockReceipt {
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
        };

        Ok(Self { eth, l1_block_info })
    }
}

#[cfg(test)]
mod tests {
    use edr_eth::{log::ExecutionLog, Bloom, Bytes};
    use edr_rpc_eth::impl_execution_receipt_tests;
    use receipt::BlockReceiptFactory;

    use super::*;
    use crate::{L1BlockInfo, OptimismChainSpec};

    impl_execution_receipt_tests! {
        OptimismChainSpec, BlockReceiptFactory {
            l1_block_info: L1BlockInfo {
                l1_base_fee: U256::from(1234),
                l1_fee_overhead: None,
                l1_base_fee_scalar: U256::from(5678),
                l1_blob_base_fee: None,
                l1_blob_base_fee_scalar: None,
            }.into(),
        } => {
            legacy, OptimismSpecId::LATEST => TypedEnvelope::Legacy(receipt::Execution::Legacy(receipt::execution::Legacy {
                root: B256::random(),
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip2930, OptimismSpecId::LATEST => TypedEnvelope::Eip2930(receipt::Execution::Eip658(receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip1559, OptimismSpecId::LATEST => TypedEnvelope::Eip2930(receipt::Execution::Eip658(receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip4844, OptimismSpecId::LATEST => TypedEnvelope::Eip4844(receipt::Execution::Eip658(receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            deposit, OptimismSpecId::LATEST => TypedEnvelope::Deposit(receipt::Execution::Deposit(receipt::execution::Deposit {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
                deposit_nonce: 0x1234,
                deposit_receipt_version: Some(1u8),
            })),
        }
    }
}
