use alloy_rlp::RlpEncodable;
pub use revm_specification::hardfork::{self, SpecId};
pub use revm_wiring::{default::block::BlockEnv, result::HaltReason};

use crate::{
    eips::eip1559::{BaseFeeParams, ConstantBaseFeeParams},
    spec::{ChainSpec, EthHeaderConstants},
    transaction,
};

/// The chain specification for Ethereum Layer 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, RlpEncodable)]
pub struct L1ChainSpec;

impl ChainSpec for L1ChainSpec {
    type BlockEnv = BlockEnv;
    type Context = ();
    type HaltReason = HaltReason;
    type Hardfork = SpecId;
    type SignedTransaction = transaction::Signed;
}

impl EthHeaderConstants for L1ChainSpec {
    const BASE_FEE_PARAMS: BaseFeeParams<Self::Hardfork> =
        BaseFeeParams::Constant(ConstantBaseFeeParams::ethereum());

    const MIN_ETHASH_DIFFICULTY: u64 = 131072;
}
