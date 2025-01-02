//! error handling and solc error codes
use std::{fmt, str::FromStr};

/// A non-exhaustive list of solidity error codes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolidityErrorCode {
    /// Warning that SPDX license identifier not provided in source file
    SpdxLicenseNotProvided,
    /// Warning: Visibility for constructor is ignored. If you want the contract
    /// to be non-deployable, making it "abstract" is sufficient
    VisibilityForConstructorIsIgnored,
    /// Warning that contract code size exceeds 24576 bytes (a limit introduced
    /// in Spurious Dragon).
    ContractExceeds24576Bytes,
    /// Warning after shanghai if init code size exceeds 49152 bytes
    ContractInitCodeSizeExceeds49152Bytes,
    /// Warning that Function state mutability can be restricted to [view,pure]
    FunctionStateMutabilityCanBeRestricted,
    /// Warning: Unused local variable
    UnusedLocalVariable,
    /// Warning: Unused function parameter. Remove or comment out the variable
    /// name to silence this warning.
    UnusedFunctionParameter,
    /// Warning: Return value of low-level calls not used.
    ReturnValueOfCallsNotUsed,
    ///  Warning: Interface functions are implicitly "virtual"
    InterfacesExplicitlyVirtual,
    /// Warning: This contract has a payable fallback function, but no receive
    /// ether function. Consider adding a receive ether function.
    PayableNoReceiveEther,
    ///  Warning: This declaration shadows an existing declaration.
    ShadowsExistingDeclaration,
    /// This declaration has the same name as another declaration.
    DeclarationSameNameAsAnother,
    /// Unnamed return variable can remain unassigned
    UnnamedReturnVariable,
    /// Unreachable code
    Unreachable,
    /// Missing pragma solidity
    PragmaSolidity,
    /// Uses transient opcodes
    TransientStorageUsed,
    /// All other error codes
    Other(u64),
}

// === impl SolidityErrorCode ===

impl SolidityErrorCode {
    /// The textual identifier for this error
    ///
    /// Returns `Err(code)` if unknown error
    pub fn as_str(&self) -> Result<&'static str, u64> {
        let s = match self {
            SolidityErrorCode::SpdxLicenseNotProvided => "license",
            SolidityErrorCode::ContractExceeds24576Bytes => "code-size",
            SolidityErrorCode::ContractInitCodeSizeExceeds49152Bytes => "init-code-size",
            SolidityErrorCode::FunctionStateMutabilityCanBeRestricted => "func-mutability",
            SolidityErrorCode::UnusedLocalVariable => "unused-var",
            SolidityErrorCode::UnusedFunctionParameter => "unused-param",
            SolidityErrorCode::ReturnValueOfCallsNotUsed => "unused-return",
            SolidityErrorCode::InterfacesExplicitlyVirtual => "virtual-interfaces",
            SolidityErrorCode::PayableNoReceiveEther => "missing-receive-ether",
            SolidityErrorCode::ShadowsExistingDeclaration => "shadowing",
            SolidityErrorCode::DeclarationSameNameAsAnother => "same-varname",
            SolidityErrorCode::UnnamedReturnVariable => "unnamed-return",
            SolidityErrorCode::Unreachable => "unreachable",
            SolidityErrorCode::PragmaSolidity => "pragma-solidity",
            SolidityErrorCode::Other(code) => return Err(*code),
            SolidityErrorCode::VisibilityForConstructorIsIgnored => "constructor-visibility",
            SolidityErrorCode::TransientStorageUsed => "transient-storage",
        };
        Ok(s)
    }
}

impl From<SolidityErrorCode> for u64 {
    fn from(code: SolidityErrorCode) -> u64 {
        match code {
            SolidityErrorCode::SpdxLicenseNotProvided => 1878,
            SolidityErrorCode::ContractExceeds24576Bytes => 5574,
            SolidityErrorCode::FunctionStateMutabilityCanBeRestricted => 2018,
            SolidityErrorCode::UnusedLocalVariable => 2072,
            SolidityErrorCode::UnusedFunctionParameter => 5667,
            SolidityErrorCode::ReturnValueOfCallsNotUsed => 9302,
            SolidityErrorCode::InterfacesExplicitlyVirtual => 5815,
            SolidityErrorCode::PayableNoReceiveEther => 3628,
            SolidityErrorCode::ShadowsExistingDeclaration => 2519,
            SolidityErrorCode::DeclarationSameNameAsAnother => 8760,
            SolidityErrorCode::UnnamedReturnVariable => 6321,
            SolidityErrorCode::Unreachable => 5740,
            SolidityErrorCode::PragmaSolidity => 3420,
            SolidityErrorCode::ContractInitCodeSizeExceeds49152Bytes => 3860,
            SolidityErrorCode::VisibilityForConstructorIsIgnored => 2462,
            SolidityErrorCode::TransientStorageUsed => 2394,
            SolidityErrorCode::Other(code) => code,
        }
    }
}

impl fmt::Display for SolidityErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Ok(name) => name.fmt(f),
            Err(code) => code.fmt(f),
        }
    }
}

impl FromStr for SolidityErrorCode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let code = match s {
            "unreachable" => SolidityErrorCode::Unreachable,
            "unused-return" => SolidityErrorCode::UnnamedReturnVariable,
            "unused-param" => SolidityErrorCode::UnusedFunctionParameter,
            "unused-var" => SolidityErrorCode::UnusedLocalVariable,
            "code-size" => SolidityErrorCode::ContractExceeds24576Bytes,
            "init-code-size" => SolidityErrorCode::ContractInitCodeSizeExceeds49152Bytes,
            "shadowing" => SolidityErrorCode::ShadowsExistingDeclaration,
            "func-mutability" => SolidityErrorCode::FunctionStateMutabilityCanBeRestricted,
            "license" => SolidityErrorCode::SpdxLicenseNotProvided,
            "pragma-solidity" => SolidityErrorCode::PragmaSolidity,
            "virtual-interfaces" => SolidityErrorCode::InterfacesExplicitlyVirtual,
            "missing-receive-ether" => SolidityErrorCode::PayableNoReceiveEther,
            "same-varname" => SolidityErrorCode::DeclarationSameNameAsAnother,
            "constructor-visibility" => SolidityErrorCode::VisibilityForConstructorIsIgnored,
            "transient-storage" => SolidityErrorCode::TransientStorageUsed,
            _ => return Err(format!("Unknown variant {s}")),
        };

        Ok(code)
    }
}

impl From<u64> for SolidityErrorCode {
    fn from(code: u64) -> Self {
        match code {
            1878 => SolidityErrorCode::SpdxLicenseNotProvided,
            5574 => SolidityErrorCode::ContractExceeds24576Bytes,
            3860 => SolidityErrorCode::ContractInitCodeSizeExceeds49152Bytes,
            2018 => SolidityErrorCode::FunctionStateMutabilityCanBeRestricted,
            2072 => SolidityErrorCode::UnusedLocalVariable,
            5667 => SolidityErrorCode::UnusedFunctionParameter,
            9302 => SolidityErrorCode::ReturnValueOfCallsNotUsed,
            5815 => SolidityErrorCode::InterfacesExplicitlyVirtual,
            3628 => SolidityErrorCode::PayableNoReceiveEther,
            2519 => SolidityErrorCode::ShadowsExistingDeclaration,
            8760 => SolidityErrorCode::DeclarationSameNameAsAnother,
            6321 => SolidityErrorCode::UnnamedReturnVariable,
            3420 => SolidityErrorCode::PragmaSolidity,
            5740 => SolidityErrorCode::Unreachable,
            2462 => SolidityErrorCode::VisibilityForConstructorIsIgnored,
            2394 => SolidityErrorCode::TransientStorageUsed,
            other => SolidityErrorCode::Other(other),
        }
    }
}
