mod builder;

use edr_evm::EthLocalBlockForChainSpec;

pub use self::builder::Builder;
use crate::OptimismChainSpec;

/// Local block type for Optimism.
pub type LocalBlock = EthLocalBlockForChainSpec<OptimismChainSpec>;
