use edr_eth::{account::AccountInfo, state::AccountStorage};

#[derive(Clone, Debug, Default)]
pub struct EdrAccount {
    pub info: AccountInfo,
    pub storage: AccountStorage,
}

impl From<AccountInfo> for EdrAccount {
    fn from(info: AccountInfo) -> Self {
        Self {
            info,
            storage: AccountStorage::default(),
        }
    }
}
