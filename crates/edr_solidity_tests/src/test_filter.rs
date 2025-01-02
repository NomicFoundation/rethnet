use std::path::Path;

/// Test filter.
pub trait TestFilter: Send + Sync {
    /// Returns whether the test should be included.
    fn matches_test(&self, test_name: &str) -> bool;

    /// Returns whether the contract should be included.
    fn matches_contract(&self, contract_name: &str) -> bool;

    /// Returns a contract with the given path should be included.
    fn matches_path(&self, path: &Path) -> bool;
}
