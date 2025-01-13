#![warn(missing_docs)]

//! The EDR EVM
//!
//! The EDR EVM exposes APIs for running and interacting with a multi-threaded
//! Ethereum Virtual Machine (or EVM).

pub use crate::{
    block::*,
    debug::{DebugContext, GetContextData},
    debug_trace::{
        debug_trace_transaction, execution_result_to_debug_result,
        register_eip_3155_and_raw_tracers_handles, register_eip_3155_tracer_handles,
        DebugTraceConfig, DebugTraceError, DebugTraceLogItem, DebugTraceResult,
        DebugTraceResultWithTraces, Eip3155AndRawTracers, TracerEip3155,
    },
    mempool::{MemPool, MemPoolAddTransactionError, OrderedTransaction},
    miner::*,
    random::RandomHashGenerator,
    runtime::{dry_run, guaranteed_dry_run, run, SyncDatabase},
};

/// Types for Ethereum blocks.
pub mod block;
/// Types for managing Ethereum blockchain
pub mod blockchain;
pub(crate) mod collections;
/// Types for configuring the runtime.
pub mod config;
mod debug;
mod debug_trace;
/// Types for interfacing with the evm.
pub mod evm;
/// Types for Ethereum hardforks
pub mod hardfork;
/// Types for managing Ethereum mem pool
pub mod mempool;
mod miner;
/// Types for managing Ethereum precompiles
pub mod precompile;
pub(crate) mod random;
/// Types for Ethereum transaction receipts.
pub mod receipt;
/// Result types for EVM execution.
pub mod result;
mod runtime;
/// Types for chain specification.
pub mod spec;
/// Database types for managing Ethereum state
pub mod state;
/// Utilities for testing
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
/// Types used for tracing EVM calls
pub mod trace;
/// Types for Ethereum transactions
pub mod transaction;
