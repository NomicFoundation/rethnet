use auto_impl::auto_impl;

use crate::{
    blockchain::SyncBlockchain,
    spec::RuntimeSpec,
    state::{DatabaseComponents, State, SyncState, WrapDatabaseRef},
};

/// Type for registering handles, specialised for EDR database component types.
pub type HandleRegister<'evm, ChainSpecT, BlockchainErrorT, DebugDataT, StateT> =
    revm::handler::register::HandleRegister<
        <ChainSpecT as RuntimeSpec>::EvmWiring<
            WrapDatabaseRef<
                DatabaseComponents<
                    &'evm dyn SyncBlockchain<
                        ChainSpecT,
                        BlockchainErrorT,
                        <StateT as State>::Error,
                    >,
                    StateT,
                >,
            >,
            DebugDataT,
        >,
    >;

/// Helper type for a chain-specific [`DebugContext`].
pub type DebugContextForChainSpec<'evm, BlockchainErrorT, ChainSpecT, DebugDataT, StateErrorT> =
    DebugContext<'evm, ChainSpecT, BlockchainErrorT, DebugDataT, Box<dyn SyncState<StateErrorT>>>;

/// Type for encapsulating contextual data and handler registration in an
/// `EvmBuilder`.
pub struct DebugContext<'evm, ChainSpecT, BlockchainErrorT, DebugDataT, StateT>
where
    ChainSpecT: RuntimeSpec,
    StateT: State,
{
    /// The contextual data.
    pub data: DebugDataT,
    /// The function to register handles.
    pub register_handles_fn: HandleRegister<'evm, ChainSpecT, BlockchainErrorT, DebugDataT, StateT>,
}

/// Trait for getting contextual data.
#[auto_impl(&mut)]
pub trait GetContextData<DataT> {
    /// Retrieves the contextual data.
    fn get_context_data(&mut self) -> &mut DataT;
}
