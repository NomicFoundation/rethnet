#![warn(missing_docs)]

//! Optimism types
//!
//! Optimism types as needed by EDR. They are based on the same primitive types
//! as `revm`.

/// Optimism RPC types
pub mod rpc;

/// Types for Optimism's EIP-2718 envelope.
pub mod eip2718;
/// Optimism harforks.
pub mod hardfork;
mod predeploy;
/// Types for Optimism receipts.
pub mod receipt;
mod spec;
pub use self::spec::OptimismChainSpec;

/// Optimism transaction types
pub mod transaction;

pub use revm_optimism::OptimismSpecId;
