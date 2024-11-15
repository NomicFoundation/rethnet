use revm_inspector::Inspector;
use revm_wiring::{
    default::{block::BlockEnv, Env, TxEnv},
    result::HaltReason,
    EthereumWiring,
};

use crate::backend::Backend;

pub type EnvWiring = Env<BlockEnv, TxEnv>;
pub type ResultAndState = revm_wiring::result::ResultAndState<HaltReason>;
