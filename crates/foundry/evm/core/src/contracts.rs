//! Commonly used contract types and functions.

use std::{collections::BTreeMap, path::PathBuf};

use alloy_json_abi::JsonAbi;
use alloy_primitives::{Address, Bytes};
use eyre::Result;
use foundry_compilers::artifacts::{CompactContractBytecode, ContractBytecodeSome};
use semver::Version;
use serde::{Deserialize, Serialize};

// Adapted from <https://github.com/foundry-rs/compilers/blob/ea346377deaf18dc1f972a06fad76df3d9aed8d9/crates/compilers/src/artifact_output/mod.rs#L45>
/// Represents compilation artifact output
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ArtifactId {
    /// The name of the contract
    pub name: String,
    /// Original source file path
    pub source: PathBuf,
    /// `solc` version that produced this artifact
    pub version: Version,
}

impl ArtifactId {
    // Copied from <https://github.com/foundry-rs/compilers/blob/ea346377deaf18dc1f972a06fad76df3d9aed8d9/crates/compilers/src/artifact_output/mod.rs#L45>
    /// Returns a `<source path>:<name>` slug that uniquely identifies an
    /// artifact
    pub fn identifier(&self) -> String {
        format!("{}:{}", self.source.to_string_lossy(), self.name)
    }
}

impl From<foundry_compilers::ArtifactId> for ArtifactId {
    fn from(value: foundry_compilers::ArtifactId) -> Self {
        let foundry_compilers::ArtifactId {
            path: _,
            name,
            source,
            version,
        } = value;

        Self {
            name,
            source,
            version,
        }
    }
}

/// Container for commonly used contract data.
#[derive(Debug, Clone)]
pub struct ContractData {
    /// Contract ABI.
    pub abi: JsonAbi,
    /// Contract creation code.
    pub bytecode: Option<Bytes>,
    /// Contract runtime code.
    pub deployed_bytecode: Option<Bytes>,
}

type ArtifactWithContractRef<'a> = (&'a ArtifactId, &'a ContractData);

/// Wrapper type that maps an artifact to a contract ABI and bytecode.
#[derive(Clone, Default, Debug)]
pub struct ContractsByArtifact(BTreeMap<ArtifactId, ContractData>);

impl ContractsByArtifact {
    /// Creates a new instance by collecting all artifacts with present bytecode
    /// from an iterator.
    ///
    /// It is recommended to use this method with an output of
    /// [`foundry_linking::Linker::get_linked_artifacts`].
    pub fn new_from_foundry_linker(
        artifacts: impl IntoIterator<Item = (foundry_compilers::ArtifactId, CompactContractBytecode)>,
    ) -> Self {
        Self(
            artifacts
                .into_iter()
                .filter_map(|(id, artifact)| {
                    let bytecode = artifact
                        .bytecode
                        .and_then(foundry_compilers::artifacts::CompactBytecode::into_bytes)?;
                    let deployed_bytecode = artifact.deployed_bytecode.and_then(
                        foundry_compilers::artifacts::CompactDeployedBytecode::into_bytes,
                    )?;

                    // Exclude artifacts with present but empty bytecode. Such artifacts are usually
                    // interfaces and abstract contracts.
                    let bytecode = (bytecode.len() > 0).then_some(bytecode);
                    let deployed_bytecode =
                        (deployed_bytecode.len() > 0).then_some(deployed_bytecode);
                    let abi = artifact.abi?;

                    Some((
                        id.into(),
                        ContractData {
                            abi,
                            bytecode,
                            deployed_bytecode,
                        },
                    ))
                })
                .collect(),
        )
    }

    /// Returns an iterator over all ids and contracts.
    pub fn iter(&self) -> impl Iterator<Item = ArtifactWithContractRef<'_>> {
        self.0.iter()
    }

    /// Get a contract by its id.
    pub fn get(&self, id: &ArtifactId) -> Option<&ContractData> {
        self.0.get(id)
    }

    /// Returns the number of contracts.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no contracts.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over the contracts.
    pub fn values(&self) -> impl Iterator<Item = &ContractData> {
        self.0.values()
    }

    /// Finds a contract which has a similar deployed bytecode as `code`.
    pub fn find_by_deployed_code(&self, code: &[u8]) -> Option<ArtifactWithContractRef<'_>> {
        self.iter().find(|(_, contract)| {
            if let Some(deployed_bytecode) = &contract.deployed_bytecode {
                bytecode_diff_score(deployed_bytecode.as_ref(), code) <= 0.1
            } else {
                false
            }
        })
    }

    /// Finds a contract which has the same contract name or identifier as `id`.
    /// If more than one is found, return error.
    pub fn find_by_name_or_identifier(
        &self,
        id: &str,
    ) -> Result<Option<ArtifactWithContractRef<'_>>> {
        let contracts = self
            .iter()
            .filter(|(artifact, _)| artifact.name == id || artifact.identifier() == id)
            .collect::<Vec<_>>();

        if contracts.len() > 1 {
            eyre::bail!("{id} has more than one implementation.");
        }

        Ok(contracts.first().cloned())
    }
}

impl From<BTreeMap<ArtifactId, ContractData>> for ContractsByArtifact {
    fn from(value: BTreeMap<ArtifactId, ContractData>) -> Self {
        Self(value)
    }
}

/// Wrapper type that maps an address to a contract identifier and contract ABI.
pub type ContractsByAddress = BTreeMap<Address, (String, JsonAbi)>;

/// Very simple fuzzy matching of contract bytecode.
///
/// Returns a value between `0.0` (identical) and `1.0` (completely different).
pub fn bytecode_diff_score<'a>(mut a: &'a [u8], mut b: &'a [u8]) -> f64 {
    // Make sure `a` is the longer one.
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }

    // Account for different lengths.
    let mut n_different_bytes = a.len() - b.len();

    // If the difference is more than 32 bytes and more than 10% of the total
    // length, we assume the bytecodes are completely different.
    // This is a simple heuristic to avoid checking every byte when the lengths are
    // very different. 32 is chosen to be a reasonable minimum as it's the size
    // of metadata hashes and one EVM word.
    if n_different_bytes > 32 && n_different_bytes * 10 > a.len() {
        return 1.0;
    }

    // Count different bytes.
    // SAFETY: `a` is longer than `b`.
    n_different_bytes += unsafe { count_different_bytes(a, b) };

    n_different_bytes as f64 / a.len() as f64
}

/// Returns the amount of different bytes between two slices.
///
/// # Safety
///
/// `a` must be at least as long as `b`.
unsafe fn count_different_bytes(a: &[u8], b: &[u8]) -> usize {
    // This could've been written as `std::iter::zip(a, b).filter(|(x, y)| x !=
    // y).count()`, however this function is very hot, and has been written to
    // be as primitive as possible for lower optimization levels.

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let len = b.len();

    let mut sum = 0;
    let mut i = 0;
    while i < len {
        // SAFETY: `a` is at least as long as `b`, and `i` is in bound of `b`.
        sum += usize::from(unsafe { *a_ptr.add(i) != *b_ptr.add(i) });
        i += 1;
    }
    sum
}

/// Artifact/Contract identifier can take the following form:
/// `<artifact file name>:<contract name>`, the `artifact file name` is the name
/// of the json file of the contract's artifact and the contract name is the
/// name of the solidity contract, like `SafeTransferLibTest.json:
/// SafeTransferLibTest`
///
/// This returns the `contract name` part
///
/// # Example
///
/// ```
/// assert_eq!(
///     "SafeTransferLibTest",
///     foundry_evm_core::contracts::get_contract_name("SafeTransferLibTest.json:SafeTransferLibTest")
/// );
/// ```
pub fn get_contract_name(id: &str) -> &str {
    id.rsplit(':').next().unwrap_or(id)
}

/// This returns the `file name` part, See [`get_contract_name`]
///
/// # Example
///
/// ```
/// assert_eq!(
///     "SafeTransferLibTest.json",
///     foundry_evm_core::contracts::get_file_name("SafeTransferLibTest.json:SafeTransferLibTest")
/// );
/// ```
pub fn get_file_name(id: &str) -> &str {
    id.split(':').next().unwrap_or(id)
}

/// Helper function to convert `CompactContractBytecode` ~>
/// `ContractBytecodeSome`
pub fn compact_to_contract(contract: CompactContractBytecode) -> Result<ContractBytecodeSome> {
    Ok(ContractBytecodeSome {
        abi: contract.abi.ok_or_else(|| eyre::eyre!("No contract abi"))?,
        bytecode: contract
            .bytecode
            .ok_or_else(|| eyre::eyre!("No contract bytecode"))?
            .into(),
        deployed_bytecode: contract
            .deployed_bytecode
            .ok_or_else(|| eyre::eyre!("No contract deployed bytecode"))?
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytecode_diffing() {
        assert_eq!(bytecode_diff_score(b"a", b"a"), 0.0);
        assert_eq!(bytecode_diff_score(b"a", b"b"), 1.0);

        let a_100 = &b"a".repeat(100)[..];
        assert_eq!(bytecode_diff_score(a_100, &b"b".repeat(100)), 1.0);
        assert_eq!(bytecode_diff_score(a_100, &b"b".repeat(99)), 1.0);
        assert_eq!(bytecode_diff_score(a_100, &b"b".repeat(101)), 1.0);
        assert_eq!(bytecode_diff_score(a_100, &b"b".repeat(120)), 1.0);
        assert_eq!(bytecode_diff_score(a_100, &b"b".repeat(1000)), 1.0);

        let a_99 = &b"a".repeat(99)[..];
        assert!(bytecode_diff_score(a_100, a_99) <= 0.01);
    }
}
