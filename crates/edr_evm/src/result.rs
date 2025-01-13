// Re-export `edr_eth` types.
pub use edr_eth::result::*;
use edr_eth::{spec::ChainSpec, transaction::TransactionValidation};

use crate::state::DatabaseComponentError;

/// EVM error type for a specific chain.
pub type EVMErrorForChain<ChainSpecT, BlockChainErrorT, StateErrorT> = EVMError<
    DatabaseComponentError<BlockChainErrorT, StateErrorT>,
    <<ChainSpecT as ChainSpec>::SignedTransaction as TransactionValidation>::ValidationError,
>;
