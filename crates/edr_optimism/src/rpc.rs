/// Types for Optimism RPC receipt.
pub mod receipt;
/// Types for Optimism RPC transaction.
pub mod transaction;

use edr_eth::{eips::eip7702, log::FilterLog, Address, Bloom, B256, U256};
use op_alloy_rpc_types::receipt::L1BlockInfo;
use serde::{Deserialize, Serialize};

/// Transaction receipt
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockReceipt {
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
    /// Deposit nonce for Optimism deposit transactions.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub deposit_nonce: Option<u64>,
    /// Deposit receipt version for Optimism deposit transactions
    ///
    /// The deposit receipt version was introduced in Canyon to indicate an
    /// update to how receipt hashes should be computed when set. The state
    /// transition process ensures this is only set for post-Canyon deposit
    /// transactions.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub deposit_receipt_version: Option<u8>,
    /// L1 block info, for non-deposit Optimism transactions.
    #[serde(flatten)]
    pub l1_block_info: L1BlockInfo,
    /// The authorization list is a list of tuples that store the address to
    /// code which the signer desires to execute in the context of their
    /// EOA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_list: Option<Vec<eip7702::SignedAuthorization>>,
}

/// Optimism RPC transaction.
#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Transaction {
    #[serde(flatten)]
    l1: edr_rpc_eth::Transaction,
    /// ECDSA recovery id
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub v: Option<u64>,
    /// Y-parity for EIP-2930 and EIP-1559 transactions. In theory these
    /// transactions types shouldn't have a `v` field, but in practice they
    /// are returned by nodes.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub y_parity: Option<bool>,
    /// ECDSA signature r
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r: Option<U256>,
    /// ECDSA signature s
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s: Option<U256>,
    /// Hash that uniquely identifies the source of the deposit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<B256>,
    /// The ETH value to mint on L2
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "alloy_serde::quantity::opt"
    )]
    pub mint: Option<u128>,
    /// Field indicating whether the transaction is a system transaction, and
    /// therefore exempt from the L2 gas limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_system_tx: Option<bool>,
}
