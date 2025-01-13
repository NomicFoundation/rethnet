// #![warn(missing_docs)]

//! NAPI bindings for EDR's core types.

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod account;
mod block;
/// Types for overriding a call.
pub mod call_override;
/// Types for casting N-API types to Rust types.
pub mod cast;
/// Supported chain types.
pub mod chains;
/// Types for configuration.
pub mod config;
/// Types related to an EDR N-API context.
pub mod context;
mod debug_trace;
/// Types for EVM execution logs.
pub mod log;
/// Types for an RPC request logger.
pub mod logger;
/// Types for Ethereum RPC providers.
pub mod provider;
/// Types for EVM execution results.
pub mod result;
/// Types relating to benchmark scenarios.
#[cfg(feature = "scenarios")]
pub mod scenarios;
/// Types for subscribing to events.
pub mod subscription;
/// Types for EVM traces.
pub mod trace;
/// Types related to Ethereum withdrawals.
pub mod withdrawal;
