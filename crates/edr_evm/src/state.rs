mod account;
mod debug;
mod diff;
mod fork;
mod irregular;
mod r#override;
mod overrides;
mod remote;
mod trie;

use std::fmt::Debug;

use auto_impl::auto_impl;
use dyn_clone::DynClone;
use edr_eth::{
    account::{Account, AccountInfo},
    Address, Bytecode, HashMap, B256, U256,
};
use edr_rpc_eth::client::RpcClientError;
use revm::DatabaseRef;
pub use revm::{
    database_interface::{Database, DatabaseCommit as StateCommit, WrapDatabaseRef},
    state::{EvmState, EvmStorage, EvmStorageSlot},
};

pub use self::{
    debug::{AccountModifierFn, StateDebug},
    diff::StateDiff,
    fork::ForkState,
    irregular::IrregularState,
    overrides::*,
    r#override::StateOverride,
    remote::RemoteState,
    trie::{AccountTrie, TrieState},
};
use crate::blockchain::BlockHash;

/// Wrapper type around a blockchain and state to implement the `Database`
/// trait.
pub struct DatabaseComponents<BlockchainT, StateT> {
    /// The blockchain component.
    pub blockchain: BlockchainT,
    /// The state component.
    pub state: StateT,
}

/// Wrapper type around a blockchain and state error.
#[derive(Debug)]
pub enum DatabaseComponentError<BlockchainErrorT, StateErrorT> {
    /// Error caused by the blockchain.
    Blockchain(BlockchainErrorT),
    /// Error caused by the state.
    State(StateErrorT),
}

impl<BlockchainT: BlockHash, StateT: State> DatabaseRef
    for DatabaseComponents<BlockchainT, StateT>
{
    type Error = DatabaseComponentError<BlockchainT::Error, StateT::Error>;

    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        self.state.basic(address).map_err(Self::Error::State)
    }

    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.state
            .code_by_hash(code_hash)
            .map_err(Self::Error::State)
    }

    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        self.state
            .storage(address, index)
            .map_err(Self::Error::State)
    }

    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        self.blockchain
            .block_hash_by_number(number)
            .map_err(Self::Error::Blockchain)
    }
}

impl<BlockchainT: BlockHash, StateT: StateCommit> StateCommit
    for DatabaseComponents<BlockchainT, StateT>
{
    fn commit(&mut self, changes: HashMap<Address, Account>) {
        self.state.commit(changes);
    }
}

/// Trait for reading state information.
#[auto_impl(&, &mut, Box, Rc, Arc)]
pub trait State {
    /// Combinatorial state error.
    type Error;

    /// Get basic account information.
    fn basic(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error>;

    /// Get account code by its hash
    fn code_by_hash(&self, code_hash: B256) -> Result<Bytecode, Self::Error>;

    /// Get storage value of address at index.
    fn storage(&self, address: Address, index: U256) -> Result<U256, Self::Error>;
}

/// Trait for reading state information.
#[auto_impl(&mut, Box)]
pub trait StateMut {
    /// Combinatorial state error.
    type Error;

    /// Get basic account information.
    fn basic_mut(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error>;

    /// Get account code by its hash
    fn code_by_hash_mut(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error>;

    /// Get storage value of address at index.
    fn storage_mut(&mut self, address: Address, index: U256) -> Result<U256, Self::Error>;
}

/// Combinatorial error for the state API
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    /// No checkpoints to revert
    #[error("No checkpoints to revert.")]
    CannotRevert,
    /// Contract with specified code hash does not exist
    #[error("Contract with code hash `{0}` does not exist.")]
    InvalidCodeHash(B256),
    /// Specified state root does not exist
    #[error("State root `{state_root:?}` does not exist (fork: {is_fork}).")]
    InvalidStateRoot {
        /// Requested state root
        state_root: B256,
        /// Whether the state root was intended for a fork
        is_fork: bool,
    },
    /// Error from the underlying RPC client
    #[error(transparent)]
    Remote(#[from] RpcClientError),
}

/// Trait that meets all requirements for a synchronous database
pub trait SyncState<E>:
    State<Error = E> + StateCommit + StateDebug<Error = E> + Debug + DynClone + Send + Sync
where
    E: Debug + Send,
{
}

impl<E> Clone for Box<dyn SyncState<E>>
where
    E: Debug + Send,
{
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

impl<S, E> SyncState<E> for S
where
    S: State<Error = E> + StateCommit + StateDebug<Error = E> + Debug + DynClone + Send + Sync,
    E: Debug + Send,
{
}
