use std::marker::PhantomData;

use derive_where::derive_where;
use dyn_clone::DynClone;
use edr_evm::{blockchain::BlockchainErrorForChainSpec, spec::RuntimeSpec};

use crate::{
    data::CallResult, debug_mine::DebugMineBlockResultForChainSpec, error::EstimateGasFailure,
    ProviderError,
};

pub trait Logger<ChainSpecT: RuntimeSpec> {
    type BlockchainError;

    /// Whether the logger is enabled.
    fn is_enabled(&self) -> bool;

    /// Sets whether the logger is enabled.
    fn set_is_enabled(&mut self, is_enabled: bool);

    fn log_call(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &CallResult<ChainSpecT::HaltReason>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _hardfork = hardfork;
        let _transaction = transaction;
        let _result = result;

        Ok(())
    }

    fn log_estimate_gas_failure(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &EstimateGasFailure<ChainSpecT::HaltReason>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _hardfork = hardfork;
        let _transaction = transaction;
        let _failure = result;

        Ok(())
    }

    fn log_interval_mined(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _hardfork = hardfork;
        let _result = result;

        Ok(())
    }

    fn log_mined_block(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _hardfork = hardfork;
        let _results = results;

        Ok(())
    }

    fn log_send_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        mining_results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _hardfork = hardfork;
        let _transaction = transaction;
        let _mining_results = mining_results;

        Ok(())
    }

    fn print_contract_decoding_error(
        &mut self,
        error: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Prints the collected logs, which correspond to the method with the
    /// provided name.
    ///
    /// Adds an empty line at the end.
    fn print_method_logs(
        &mut self,
        method: &str,
        error: Option<&ProviderError<ChainSpecT>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

pub trait SyncLogger<ChainSpecT: RuntimeSpec>: Logger<ChainSpecT> + DynClone + Send + Sync {}

impl<ChainSpecT, T> SyncLogger<ChainSpecT> for T
where
    ChainSpecT: RuntimeSpec,
    T: Logger<ChainSpecT> + DynClone + Send + Sync,
{
}

impl<ChainSpecT: RuntimeSpec, BlockchainErrorT> Clone
    for Box<dyn SyncLogger<ChainSpecT, BlockchainError = BlockchainErrorT>>
{
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

/// A logger that does nothing.
#[derive_where(Clone, Default)]
pub struct NoopLogger<ChainSpecT: RuntimeSpec> {
    _phantom: PhantomData<ChainSpecT>,
}

impl<ChainSpecT: RuntimeSpec> Logger<ChainSpecT> for NoopLogger<ChainSpecT> {
    type BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>;

    fn is_enabled(&self) -> bool {
        false
    }

    fn set_is_enabled(&mut self, _is_enabled: bool) {}

    fn print_contract_decoding_error(
        &mut self,
        _error: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn print_method_logs(
        &mut self,
        _method: &str,
        _error: Option<&ProviderError<ChainSpecT>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}
