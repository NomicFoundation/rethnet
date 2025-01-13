use edr_eth::{
    eips::{eip2718::TypedEnvelope, eip7702},
    l1,
    log::FilterLog,
    receipt::{
        self, AsExecutionReceipt as _, Execution, ExecutionReceipt as _, TransactionReceipt,
    },
    transaction::{self, TransactionType as _},
    Address, Bloom, B256, U256,
};
use serde::{Deserialize, Serialize};

use crate::RpcTypeFrom;

/// Transaction receipt
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Block {
    /// Hash of the block this transaction was included within.
    pub block_hash: B256,
    /// Number of the block this transaction was included within.
    #[serde(default, with = "alloy_serde::quantity")]
    pub block_number: u64,
    /// Transaction Hash.
    pub transaction_hash: B256,
    /// Index within the block.
    #[serde(default, with = "alloy_serde::quantity")]
    pub transaction_index: u64,
    /// Transaction type.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt",
        rename = "type"
    )]
    pub transaction_type: Option<u8>,
    /// Address of the sender
    pub from: Address,
    /// Address of the receiver. None when its a contract creation transaction.
    pub to: Option<Address>,
    /// The sum of gas used by this transaction and all preceding transactions
    /// in the same block.
    #[serde(with = "alloy_serde::quantity")]
    pub cumulative_gas_used: u64,
    /// Gas used by this transaction alone.
    #[serde(with = "alloy_serde::quantity")]
    pub gas_used: u64,
    /// Contract address created, or None if not a deployment.
    pub contract_address: Option<Address>,
    /// Logs generated within this transaction
    pub logs: Vec<FilterLog>,
    /// Bloom filter of the logs generated within this transaction
    pub logs_bloom: Bloom,
    /// The post-transaction stateroot (pre-Byzantium)
    ///
    /// EIP98 makes this optional field, if it's missing then skip serializing
    /// it
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "root")]
    pub state_root: Option<B256>,
    /// Status code indicating whether the transaction executed successfully
    /// (post-Byzantium)
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub status: Option<bool>,
    /// The price paid post-execution by the transaction (i.e. base fee +
    /// priority fee). Both fields in 1559-style transactions are maximums
    /// (max fee + max priority fee), the amount that's actually paid by
    /// users can only be determined post-execution
    // #[serde(with = "alloy_serde::quantity::opt")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_gas_price: Option<U256>,
    /// The authorization list is a list of tuples that store the address to
    /// code which the signer desires to execute in the context of their
    /// EOA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_list: Option<Vec<eip7702::SignedAuthorization>>,
}

impl RpcTypeFrom<receipt::BlockReceipt<TypedEnvelope<receipt::Execution<FilterLog>>>> for Block {
    type Hardfork = l1::SpecId;

    fn rpc_type_from(
        value: &receipt::BlockReceipt<TypedEnvelope<receipt::Execution<FilterLog>>>,
        hardfork: Self::Hardfork,
    ) -> Self {
        let transaction_type = if hardfork >= l1::SpecId::BERLIN {
            Some(u8::from(value.inner.transaction_type()))
        } else {
            None
        };

        Self {
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
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Legacy transaction is missing state root or status")]
    MissingStateRootOrStatus,
    #[error("Missing status")]
    MissingStatus,
    #[error("Unknown transaction type: {0}")]
    UnknownType(u8),
}

impl TryFrom<Block> for receipt::BlockReceipt<TypedEnvelope<receipt::Execution<FilterLog>>> {
    type Error = ConversionError;

    fn try_from(value: Block) -> Result<Self, Self::Error> {
        let transaction_type = value
            .transaction_type
            .map_or(Ok(transaction::Type::Legacy), transaction::Type::try_from)
            .map_err(ConversionError::UnknownType)?;

        let execution = if transaction_type == transaction::Type::Legacy {
            if let Some(status) = value.status {
                Execution::Eip658(receipt::execution::Eip658 {
                    status,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                })
            } else if let Some(state_root) = value.state_root {
                Execution::Legacy(receipt::execution::Legacy {
                    root: state_root,
                    cumulative_gas_used: value.cumulative_gas_used,
                    logs_bloom: value.logs_bloom,
                    logs: value.logs,
                })
            } else {
                return Err(ConversionError::MissingStateRootOrStatus);
            }
        } else {
            Execution::Eip658(receipt::execution::Eip658 {
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

#[cfg(test)]
mod test {
    use assert_json_diff::assert_json_eq;
    use edr_eth::{
        eips::eip2718::TypedEnvelope,
        l1::{self, L1ChainSpec},
        log::ExecutionLog,
        Bloom, Bytes,
    };
    use edr_evm::block::EthBlockReceiptFactory;
    use serde_json::json;

    use crate::{impl_execution_receipt_tests, receipt};

    #[test]
    fn test_matches_hardhat_serialization() -> anyhow::Result<()> {
        // Generated with the "Hardhat Network provider eth_getTransactionReceipt should
        // return the right values for successful txs" hardhat-core test.
        let receipt_from_hardhat = json!({
          "transactionHash": "0x08d14db1a6253234f7efc94fc661f52b708882552af37ebf4f5cd904618bb208",
          "transactionIndex": "0x0",
          "blockHash": "0x404b3b3ed507ff47178e9ca9d7757165050180091e1cc17de7981871a6e5785a",
          "blockNumber": "0x2",
          "from": "0xbe862ad9abfe6f22bcb087716c7d89a26051f74c",
          "to": "0x61de9dc6f6cff1df2809480882cfd3c2364b28f7",
          "cumulativeGasUsed": "0xaf91",
          "gasUsed": "0xaf91",
          "contractAddress": null,
          "logs": [
            {
              "removed": false,
              "logIndex": "0x0",
              "transactionIndex": "0x0",
              "transactionHash": "0x08d14db1a6253234f7efc94fc661f52b708882552af37ebf4f5cd904618bb208",
              "blockHash": "0x404b3b3ed507ff47178e9ca9d7757165050180091e1cc17de7981871a6e5785a",
              "blockNumber": "0x2",
              "address": "0x61de9dc6f6cff1df2809480882cfd3c2364b28f7",
              "data": "0x000000000000000000000000000000000000000000000000000000000000000a",
              "topics": [
                "0x3359f789ea83a10b6e9605d460de1088ff290dd7b3c9a155c896d45cf495ed4d",
                "0x0000000000000000000000000000000000000000000000000000000000000000"
              ]
            }
          ],
          "logsBloom":
            "0x00000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000400000000000000000020000000000000000000800000002000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000000",
          "type": "0x2",
          "status": "0x1",
          "effectiveGasPrice": "0x699e6346"
        });

        let deserialized: receipt::Block = serde_json::from_value(receipt_from_hardhat.clone())?;

        let serialized = serde_json::to_value(deserialized)?;
        assert_json_eq!(receipt_from_hardhat, serialized);

        Ok(())
    }

    impl_execution_receipt_tests! {
        L1ChainSpec, EthBlockReceiptFactory::default() => {
            legacy, l1::SpecId::LATEST => TypedEnvelope::Legacy(edr_eth::receipt::Execution::Legacy(edr_eth::receipt::execution::Legacy {
                root: B256::random(),
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip2930, l1::SpecId::LATEST => TypedEnvelope::Eip2930(edr_eth::receipt::Execution::Eip658(edr_eth::receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip1559, l1::SpecId::LATEST => TypedEnvelope::Eip2930(edr_eth::receipt::Execution::Eip658(edr_eth::receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
            eip658_eip4844, l1::SpecId::LATEST => TypedEnvelope::Eip4844(edr_eth::receipt::Execution::Eip658(edr_eth::receipt::execution::Eip658 {
                status: true,
                cumulative_gas_used: 0xffff,
                logs_bloom: Bloom::random(),
                logs: vec![
                    ExecutionLog::new_unchecked(Address::random(), vec![B256::random(), B256::random()], Bytes::new()),
                    ExecutionLog::new_unchecked(Address::random(), Vec::new(), Bytes::from_static(b"test"))
                ],
            })),
        }
    }
}
