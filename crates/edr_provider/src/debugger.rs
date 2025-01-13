use core::fmt::Debug;

use edr_eth::spec::HaltReasonTrait;
use edr_evm::{
    evm::handler::register::EvmHandler,
    spec::EvmWiring,
    state::Database,
    trace::{register_trace_collector_handles, TraceCollector},
    GetContextData,
};

use crate::{
    console_log::{register_console_log_handles, ConsoleLogCollector},
    mock::{register_mocking_handles, Mocker},
};

/// Registers debugger handles.
pub fn register_debugger_handles<EvmWiringT>(handler: &mut EvmHandler<'_, EvmWiringT>)
where
    EvmWiringT: EvmWiring<
        ExternalContext: GetContextData<ConsoleLogCollector>
                             + GetContextData<Mocker>
                             + GetContextData<TraceCollector<EvmWiringT::HaltReason>>,
        Database: Database<Error: Debug>,
    >,
{
    register_console_log_handles(handler);
    register_mocking_handles(handler);
    register_trace_collector_handles(handler);
}

pub struct Debugger<HaltReasonT: HaltReasonTrait> {
    pub console_logger: ConsoleLogCollector,
    pub mocker: Mocker,
    pub trace_collector: TraceCollector<HaltReasonT>,
}

impl<HaltReasonT: HaltReasonTrait> Debugger<HaltReasonT> {
    /// Creates a new instance with the provided mocker.
    /// If verbose is true, full stack and memory will be recorded for each
    /// step.
    pub fn with_mocker(mocker: Mocker, verbose: bool) -> Self {
        Self {
            console_logger: ConsoleLogCollector::default(),
            mocker,
            trace_collector: TraceCollector::new(verbose),
        }
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<ConsoleLogCollector> for Debugger<HaltReasonT> {
    fn get_context_data(&mut self) -> &mut ConsoleLogCollector {
        &mut self.console_logger
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<Mocker> for Debugger<HaltReasonT> {
    fn get_context_data(&mut self) -> &mut Mocker {
        &mut self.mocker
    }
}

impl<HaltReasonT: HaltReasonTrait> GetContextData<TraceCollector<HaltReasonT>>
    for Debugger<HaltReasonT>
{
    fn get_context_data(&mut self) -> &mut TraceCollector<HaltReasonT> {
        &mut self.trace_collector
    }
}
