use derive_where::derive_where;
use edr_eth::{account::AccountInfo, hash_map::Entry, Address, Bytecode, HashMap, B256, U256};
use edr_rpc_eth::spec::RpcSpec;

use super::RemoteState;
use crate::state::{account::EdrAccount, State, StateError, StateMut};

/// A cached version of [`RemoteState`].
#[derive_where(Debug)]
pub struct CachedRemoteState<ChainSpecT: RpcSpec> {
    remote: RemoteState<ChainSpecT>,
    /// Mapping of block numbers to cached accounts
    account_cache: HashMap<u64, HashMap<Address, EdrAccount>>,
    /// Mapping of block numbers to cached code
    code_cache: HashMap<u64, HashMap<B256, Bytecode>>,
}

impl<ChainSpecT: RpcSpec> CachedRemoteState<ChainSpecT> {
    /// Constructs a new [`CachedRemoteState`].
    pub fn new(remote: RemoteState<ChainSpecT>) -> Self {
        Self {
            remote,
            account_cache: HashMap::new(),
            code_cache: HashMap::new(),
        }
    }
}

impl<ChainSpecT: RpcSpec> StateMut for CachedRemoteState<ChainSpecT> {
    type Error = StateError;

    fn basic_mut(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let block_accounts = self
            .account_cache
            .entry(self.remote.block_number())
            .or_default();

        if let Some(account) = block_accounts.get(&address) {
            return Ok(Some(account.info.clone()));
        }

        if let Some(account_info) =
            fetch_remote_account(address, &self.remote, &mut self.code_cache)?
        {
            if self.remote.is_cacheable()? {
                block_accounts.insert(address, account_info.clone().into());
            }
            return Ok(Some(account_info));
        }

        Ok(None)
    }

    fn code_by_hash_mut(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        let block_code = self
            .code_cache
            .entry(self.remote.block_number())
            .or_default();

        block_code
            .get(&code_hash)
            .cloned()
            .ok_or(StateError::InvalidCodeHash(code_hash))
    }

    fn storage_mut(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let block_accounts = self
            .account_cache
            .entry(self.remote.block_number())
            .or_default();

        Ok(match block_accounts.entry(address) {
            Entry::Occupied(mut account_entry) => {
                match account_entry.get_mut().storage.entry(index) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let value = self.remote.storage(address, index)?;
                        if self.remote.is_cacheable()? {
                            *entry.insert(value)
                        } else {
                            value
                        }
                    }
                }
            }
            Entry::Vacant(account_entry) => {
                // account needs to be loaded for us to access slots.
                let mut account =
                    fetch_remote_account(address, &self.remote, &mut self.code_cache)?
                        .map_or_else(EdrAccount::default, EdrAccount::from);

                let value = self.remote.storage(address, index)?;

                if self.remote.is_cacheable()? {
                    account.storage.insert(index, value);
                    account_entry.insert(account);
                }

                value
            }
        })
    }
}

/// Fetches an account from the remote state. If it exists, code is split off
/// and stored separately in the provided cache.
fn fetch_remote_account<ChainSpecT: RpcSpec>(
    address: Address,
    remote: &RemoteState<ChainSpecT>,
    code_cache: &mut HashMap<u64, HashMap<B256, Bytecode>>,
) -> Result<Option<AccountInfo>, StateError> {
    let account = remote.basic(address)?.map(|mut account_info| {
        // Always cache code regardless of the block number for two reasons:
        // 1. It's an invariant of this trait getting an `AccountInfo` by calling
        //    `basic`,
        // one can call `code_by_hash` with `AccountInfo.code_hash` and get the code.
        // 2. Since the code is identified by its hash, it never goes stale.
        if let Some(code) = account_info.code.take() {
            let block_code = code_cache.entry(remote.block_number()).or_default();

            block_code.entry(account_info.code_hash).or_insert(code);
        }
        account_info
    });

    Ok(account)
}

#[cfg(all(test, feature = "test-remote"))]
mod tests {
    use std::{str::FromStr, sync::Arc};

    use edr_eth::l1::L1ChainSpec;
    use edr_rpc_eth::client::EthRpcClient;
    use edr_test_utils::env::get_alchemy_url;
    use tokio::runtime;

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn no_cache_for_unsafe_block_number() {
        let tempdir = tempfile::tempdir().expect("can create tempdir");

        let rpc_client = EthRpcClient::<L1ChainSpec>::new(
            &get_alchemy_url(),
            tempdir.path().to_path_buf(),
            None,
        )
        .expect("url ok");

        let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
            .expect("failed to parse address");

        // Latest block number is always unsafe
        let block_number = rpc_client.block_number().await.unwrap();

        let runtime = runtime::Handle::current();

        let remote = RemoteState::new(runtime, Arc::new(rpc_client), block_number);
        let mut cached = CachedRemoteState::new(remote);

        let account_info = cached
            .basic_mut(dai_address)
            .expect("should succeed")
            .unwrap();

        cached
            .storage_mut(dai_address, U256::from(0))
            .expect("should succeed");

        for entry in cached.account_cache.values() {
            assert!(entry.is_empty());
        }

        cached
            .code_by_hash_mut(account_info.code_hash)
            .expect("should succeed");
    }
}
