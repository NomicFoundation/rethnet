#![warn(missing_docs)]

//! Optimism types
//!
//! Optimism types as needed by EDR. They are based on the same primitive types
//! as `revm`.

/// Optimism RPC types
pub mod rpc;

/// Types for Optimism blocks.
pub mod block;
/// Types for Optimism's EIP-2718 envelope.
pub mod eip2718;
/// Optimism harforks.
pub mod hardfork;
/// Types for Optimism receipts.
pub mod receipt;
mod spec;
pub use self::spec::OptimismChainSpec;

/// Optimism transaction types
pub mod transaction;

use edr_eth::U256;
pub use revm_optimism::OptimismSpecId;

/// Helper type for constructing an [`revm_optimism::L1BlockInfo`].
///
/// This type duplicates [`revm_optimism::L1BlockInfo`] but excludes the private
/// field to allow manual construction.
pub struct L1BlockInfo {
    /// The base fee of the L1 origin block.
    pub l1_base_fee: U256,
    /// The current L1 fee overhead. None if Ecotone is activated.
    pub l1_fee_overhead: Option<U256>,
    /// The current L1 fee scalar.
    pub l1_base_fee_scalar: U256,
    /// The current L1 blob base fee. None if Ecotone is not activated, except
    /// if `empty_scalars` is `true`.
    pub l1_blob_base_fee: Option<U256>,
    /// The current L1 blob base fee scalar. None if Ecotone is not activated.
    pub l1_blob_base_fee_scalar: Option<U256>,
}

impl From<L1BlockInfo> for revm_optimism::L1BlockInfo {
    fn from(value: L1BlockInfo) -> Self {
        // `revm_optimism::L1BlockInfo` contains a private field, so we need to
        // construct it like this
        let mut l1_block_info = Self::default();
        l1_block_info.l1_base_fee = value.l1_base_fee;
        l1_block_info.l1_fee_overhead = value.l1_fee_overhead;
        l1_block_info.l1_base_fee_scalar = value.l1_base_fee_scalar;
        l1_block_info.l1_blob_base_fee = value.l1_blob_base_fee;
        l1_block_info.l1_blob_base_fee_scalar = value.l1_blob_base_fee_scalar;

        l1_block_info
    }
}
