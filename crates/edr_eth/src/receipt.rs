// Part of this code was adapted from foundry and is distributed under their
// licenss:
// - https://github.com/foundry-rs/foundry/blob/01b16238ff87dc7ca8ee3f5f13e389888c2a2ee4/LICENSE-APACHE
// - https://github.com/foundry-rs/foundry/blob/01b16238ff87dc7ca8ee3f5f13e389888c2a2ee4/LICENSE-MIT
// For the original context see: https://github.com/foundry-rs/foundry/blob/01b16238ff87dc7ca8ee3f5f13e389888c2a2ee4/anvil/core/src/eth/receipt.rs

#![allow(missing_docs)]

mod block;
/// Types for execution receipts.
pub mod execution;
mod factory;
mod transaction;

use auto_impl::auto_impl;

pub use self::{block::BlockReceipt, factory::ReceiptFactory, transaction::TransactionReceipt};
use crate::{Address, Bloom, B256, U256};

/// Log generated after execution of a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(untagged))]
pub enum Execution<LogT> {
    /// Legacy receipt.
    Legacy(self::execution::Legacy<LogT>),
    /// EIP-658 receipt.
    Eip658(self::execution::Eip658<LogT>),
}

/// Type representing either the state root (pre-EIP-658) or the status code
/// (post-EIP-658).
#[derive(Debug, PartialEq, Eq)]
pub enum RootOrStatus<'root> {
    /// State root (pre-EIP-658).
    Root(&'root B256),
    /// Status code (post-EIP-658).
    Status(bool),
}

/// Trait for a receipt that internally contains an execution receipt.
pub trait AsExecutionReceipt {
    /// The type of the inner execution receipt.
    type ExecutionReceipt: ExecutionReceipt;

    /// Returns a reference to the inner execution receipt.
    fn as_execution_receipt(&self) -> &Self::ExecutionReceipt;
}

/// Trait for a receipt that's generated after execution of a transaction.
#[auto_impl(Box, Arc)]
pub trait ExecutionReceipt {
    type Log;

    /// Returns the cumulative gas used in the block after this transaction was
    /// executed.
    fn cumulative_gas_used(&self) -> u64;
    /// Returns the bloom filter of the logs generated within this transaction.
    fn logs_bloom(&self) -> &Bloom;
    /// Returns the logs generated within this transaction.
    fn transaction_logs(&self) -> &[Self::Log];
    /// Returns the state root (pre-EIP-658) or status (post-EIP-658) of the
    /// receipt.
    fn root_or_status(&self) -> RootOrStatus<'_>;
}

pub trait MapReceiptLogs<OldLogT, NewLogT, OutputT> {
    /// Maps the logs of the receipt to a new type.
    fn map_logs(self, map_fn: impl FnMut(OldLogT) -> NewLogT) -> OutputT;
}

#[auto_impl(Box, Arc)]
pub trait ReceiptTrait {
    fn block_number(&self) -> u64;

    fn block_hash(&self) -> &B256;

    fn contract_address(&self) -> Option<&Address>;

    fn effective_gas_price(&self) -> Option<&U256>;

    fn from(&self) -> &Address;

    fn gas_used(&self) -> u64;

    fn to(&self) -> Option<&Address>;

    /// Returns the transaction hash.
    fn transaction_hash(&self) -> &B256;

    fn transaction_index(&self) -> u64;
}
