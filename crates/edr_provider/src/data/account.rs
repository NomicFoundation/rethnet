use edr_eth::{
    account::{Account, AccountInfo, AccountStatus},
    signature::public_key_to_address,
    spec::HardforkTrait,
    Address, HashMap, KECCAK_EMPTY,
};
use indexmap::IndexMap;

use crate::config::{self, Provider};

pub(super) struct InitialAccounts {
    pub local_accounts: IndexMap<Address, k256::SecretKey>,
    pub genesis_state: HashMap<Address, Account>,
}

pub(super) fn create_accounts<HardforkT: HardforkTrait>(
    config: &Provider<HardforkT>,
) -> InitialAccounts {
    let mut local_accounts = IndexMap::default();

    let genesis_state = config
        .accounts
        .iter()
        .map(
            |config::OwnedAccount {
                 secret_key,
                 balance,
             }| {
                let address = public_key_to_address(secret_key.public_key());
                let genesis_account = AccountInfo {
                    balance: *balance,
                    nonce: 0,
                    code: None,
                    code_hash: KECCAK_EMPTY,
                };

                local_accounts.insert(address, secret_key.clone());

                (address, config::Account::from(genesis_account))
            },
        )
        .chain(config.genesis_state.clone())
        .map(|(address, config::Account { info, storage })| {
            let account = Account {
                info,
                storage,
                status: AccountStatus::Created | AccountStatus::Touched,
            };

            (address, account)
        })
        .collect();

    InitialAccounts {
        local_accounts,
        genesis_state,
    }
}
