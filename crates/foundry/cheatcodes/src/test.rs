//! Implementations of [`Testing`](crate::Group::Testing) cheatcodes.

use alloy_sol_types::SolValue;
use foundry_evm_core::constants::{MAGIC_ASSUME, MAGIC_SKIP};

use crate::{
    Cheatcode, Cheatcodes, CheatsCtxt, DatabaseExt, Error, Result,
    Vm::{assumeCall, rpcUrlCall, rpcUrlStructsCall, rpcUrlsCall, skipCall, sleepCall},
};

pub(crate) mod assert;
pub(crate) mod expect;

impl Cheatcode for assumeCall {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { condition } = self;
        if *condition {
            Ok(Vec::default())
        } else {
            Err(Error::from(MAGIC_ASSUME))
        }
    }
}

impl Cheatcode for rpcUrlCall {
    fn apply(&self, state: &mut Cheatcodes) -> Result {
        let Self { rpcAlias } = self;
        state.config.rpc_url(rpcAlias).map(|url| url.abi_encode())
    }
}

impl Cheatcode for rpcUrlsCall {
    fn apply(&self, state: &mut Cheatcodes) -> Result {
        let Self {} = self;
        state.config.rpc_urls().map(|urls| urls.abi_encode())
    }
}

impl Cheatcode for rpcUrlStructsCall {
    fn apply(&self, state: &mut Cheatcodes) -> Result {
        let Self {} = self;
        state.config.rpc_urls().map(|urls| urls.abi_encode())
    }
}

impl Cheatcode for sleepCall {
    fn apply(&self, _state: &mut Cheatcodes) -> Result {
        let Self { duration } = self;
        let sleep_duration = std::time::Duration::from_millis(duration.saturating_to());
        std::thread::sleep(sleep_duration);
        Ok(Vec::default())
    }
}

impl Cheatcode for skipCall {
    fn apply_full<DB: DatabaseExt>(&self, ccx: &mut CheatsCtxt<DB>) -> Result {
        let Self { skipTest } = *self;
        if skipTest {
            // Skip should not work if called deeper than at test level.
            // Since we're not returning the magic skip bytes, this will cause a test
            // failure.
            ensure!(
                ccx.ecx.journaled_state.depth() <= 1,
                "`skip` can only be used at test level"
            );
            Err(MAGIC_SKIP.into())
        } else {
            Ok(Vec::default())
        }
    }
}
