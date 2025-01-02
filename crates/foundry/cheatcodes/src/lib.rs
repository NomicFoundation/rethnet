//! # foundry-cheatcodes
//!
//! Foundry cheatcodes implementations.

#![warn(
    missing_docs,
    unreachable_pub,
    unused_crate_dependencies,
    rust_2018_idioms
)]
#![allow(elided_lifetimes_in_paths)] // Cheats context uses 3 lifetimes

#[macro_use]
pub extern crate foundry_cheatcodes_spec as spec;
#[macro_use]
extern crate tracing;

use alloy_primitives::Address;
pub use config::{CheatsConfig, CheatsConfigOptions, ExecutionContextConfig};
pub use endpoints::{RpcEndpoint, RpcEndpoints};
pub use error::{Error, ErrorKind, Result};
use foundry_evm_core::backend::DatabaseExt;
pub use fs_permissions::{FsAccessKind, FsAccessPermission, FsPermissions, PathPermission};
pub use inspector::{BroadcastableTransaction, BroadcastableTransactions, Cheatcodes, Context};
use revm::{ContextPrecompiles, InnerEvmContext};
pub use spec::{CheatcodeDef, Vm};

mod ens;
#[macro_use]
mod error;
mod base64;
mod cache;
mod config;
mod endpoints;
mod env;
mod evm;
mod fs;
mod fs_permissions;
mod inspector;
mod json;
mod string;
mod test;
mod toml;
mod utils;

pub use cache::{CachedChains, CachedEndpoints, StorageCachingConfig};
pub use test::expect::ExpectedCallTracker;
pub use Vm::ExecutionContext;

/// Cheatcode implementation.
pub(crate) trait Cheatcode: CheatcodeDef + DynCheatcode {
    /// Applies this cheatcode to the given state.
    ///
    /// Implement this function if you don't need access to the EVM data.
    #[allow(clippy::unimplemented)]
    fn apply(&self, state: &mut Cheatcodes) -> Result {
        let _ = state;
        unimplemented!("{}", Self::CHEATCODE.func.id)
    }

    /// Applies this cheatcode to the given context.
    ///
    /// Implement this function if you need access to the EVM data.
    #[inline(always)]
    fn apply_full<DB: DatabaseExt>(&self, ccx: &mut CheatsCtxt<DB>) -> Result {
        self.apply(ccx.state)
    }

    #[inline]
    fn apply_traced<DB: DatabaseExt>(&self, ccx: &mut CheatsCtxt<DB>) -> Result {
        // Separate and non-generic functions to avoid inline and monomorphization
        // bloat.
        #[inline(never)]
        fn trace_span_and_call(cheat: &dyn DynCheatcode) -> tracing::span::EnteredSpan {
            let span = debug_span!(target: "cheatcodes", "apply");
            if !span.is_disabled() {
                if enabled!(tracing::Level::TRACE) {
                    span.record("cheat", tracing::field::debug(cheat.as_debug()));
                } else {
                    span.record("id", cheat.cheatcode().func.id);
                }
            }
            let entered = span.entered();
            trace!(target: "cheatcodes", "applying");
            entered
        }

        #[inline(never)]
        fn trace_return(result: &Result) {
            trace!(
                target: "cheatcodes",
                return = match result {
                    Ok(b) => hex::encode(b),
                    Err(e) => e.to_string(),
                }
            );
        }

        let _span = trace_span_and_call(self);
        let result = self.apply_full(ccx);
        trace_return(&result);
        result
    }
}

pub(crate) trait DynCheatcode {
    fn cheatcode(&self) -> &'static foundry_cheatcodes_spec::Cheatcode<'static>;
    fn as_debug(&self) -> &dyn std::fmt::Debug;
}

impl<T: Cheatcode> DynCheatcode for T {
    fn cheatcode(&self) -> &'static foundry_cheatcodes_spec::Cheatcode<'static> {
        T::CHEATCODE
    }

    fn as_debug(&self) -> &dyn std::fmt::Debug {
        self
    }
}

/// The cheatcode context, used in [`Cheatcode`].
pub(crate) struct CheatsCtxt<'cheats, 'evm, DB: DatabaseExt> {
    /// The cheatcodes inspector state.
    pub(crate) state: &'cheats mut Cheatcodes,
    /// The EVM data.
    pub(crate) ecx: &'evm mut InnerEvmContext<DB>,
    /// The precompiles context.
    pub(crate) precompiles: &'evm mut ContextPrecompiles<DB>,
    /// The original `msg.sender`.
    pub(crate) caller: Address,
}

impl<'cheats, 'evm, DB: DatabaseExt> std::ops::Deref for CheatsCtxt<'cheats, 'evm, DB> {
    type Target = InnerEvmContext<DB>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.ecx
    }
}

impl<'cheats, 'evm, DB: DatabaseExt> std::ops::DerefMut for CheatsCtxt<'cheats, 'evm, DB> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.ecx
    }
}

impl<'cheats, 'evm, DB: DatabaseExt> CheatsCtxt<'cheats, 'evm, DB> {
    #[inline]
    pub(crate) fn is_precompile(&self, address: &Address) -> bool {
        self.precompiles.contains(address)
    }
}
