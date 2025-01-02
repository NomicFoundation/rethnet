use napi_derive::napi;

/// A compilation artifact.
#[derive(Clone, Debug)]
#[napi(object)]
pub struct Artifact {
    /// The identifier of the artifact.
    pub id: ArtifactId,
    /// The test contract.
    pub contract: ContractData,
}

/// The identifier of a Solidity contract.
#[derive(Clone, Debug)]
#[napi(object)]
pub struct ArtifactId {
    /// The name of the contract.
    pub name: String,
    /// Original source file path.
    pub source: String,
    /// The solc semver string.
    pub solc_version: String,
}

impl From<edr_solidity_tests::contracts::ArtifactId> for ArtifactId {
    fn from(value: edr_solidity_tests::contracts::ArtifactId) -> Self {
        Self {
            name: value.name,
            source: value.source.to_string_lossy().to_string(),
            solc_version: value.version.to_string(),
        }
    }
}

impl TryFrom<ArtifactId> for edr_solidity_tests::contracts::ArtifactId {
    type Error = napi::Error;

    fn try_from(value: ArtifactId) -> napi::Result<Self> {
        Ok(edr_solidity_tests::contracts::ArtifactId {
            name: value.name,
            source: value.source.parse().map_err(|_err| {
                napi::Error::new(napi::Status::GenericFailure, "Invalid source path")
            })?,
            version: value.solc_version.parse().map_err(|_err| {
                napi::Error::new(napi::Status::GenericFailure, "Invalid solc semver string")
            })?,
        })
    }
}

/// A test contract to execute.
#[derive(Clone, Debug)]
#[napi(object)]
pub struct ContractData {
    /// Contract ABI as json string.
    pub abi: String,
    /// Contract creation code as hex string. It can be missing if the contract
    /// is ABI only.
    pub bytecode: Option<String>,
    /// Contract runtime code as hex string. It can be missing if the contract
    /// is ABI only.
    pub deployed_bytecode: Option<String>,
}

impl TryFrom<ContractData> for edr_solidity_tests::contracts::ContractData {
    type Error = napi::Error;

    fn try_from(contract: ContractData) -> napi::Result<Self> {
        Ok(edr_solidity_tests::contracts::ContractData {
            abi: serde_json::from_str(&contract.abi).map_err(|_err| {
                napi::Error::new(napi::Status::GenericFailure, "Invalid JSON ABI")
            })?,
            bytecode: contract
                .bytecode
                .map(|b| {
                    b.parse().map_err(|_err| {
                        napi::Error::new(
                            napi::Status::GenericFailure,
                            "Invalid hex bytecode for contract",
                        )
                    })
                })
                .transpose()?,
            deployed_bytecode: contract
                .deployed_bytecode
                .map(|b| {
                    b.parse().map_err(|_err| {
                        napi::Error::new(
                            napi::Status::GenericFailure,
                            "Invalid hex deployed bytecode for contract",
                        )
                    })
                })
                .transpose()?,
        })
    }
}
