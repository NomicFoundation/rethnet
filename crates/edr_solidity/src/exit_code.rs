//! Exit code of the EVM.

use edr_eth::{
    result::{HaltReason, OutOfGasError},
    spec::HaltReasonTrait,
};

/// Represents the exit code of the EVM.
#[derive(Clone, Debug)]
pub enum ExitCode<HaltReasonT: HaltReasonTrait> {
    /// Execution was successful.
    Success,
    /// Execution was reverted.
    Revert,
    /// Indicates that the EVM has experienced an exceptional halt.
    Halt(HaltReasonT),
}

impl<HaltReasonT: HaltReasonTrait> ExitCode<HaltReasonT> {
    /// Returns whether the exit code is an error.
    pub fn is_error(&self) -> bool {
        !matches!(self, Self::Success)
    }

    /// Returns whether the exit code is a contract too large error.
    pub fn is_contract_too_large_error(&self) -> bool {
        if let Self::Halt(reason) = self {
            *reason == HaltReason::CreateContractSizeLimit.into()
        } else {
            false
        }
    }

    /// Returns whether the exit code is an invalid opcode error.
    pub fn is_invalid_opcode_error(&self) -> bool {
        if let Self::Halt(reason) = self {
            (*reason == HaltReason::InvalidFEOpcode.into())
                | (*reason == HaltReason::OpcodeNotFound.into())
                | (*reason == HaltReason::NotActivated.into())
        } else {
            false
        }
    }

    /// Returns whether the exit code is an out of gas error.
    pub fn is_out_of_gas_error(&self) -> bool {
        if let Self::Halt(reason) = self {
            (*reason == HaltReason::OutOfGas(OutOfGasError::Basic).into())
                | (*reason == HaltReason::OutOfGas(OutOfGasError::MemoryLimit).into())
                | (*reason == HaltReason::OutOfGas(OutOfGasError::Memory).into())
                | (*reason == HaltReason::OutOfGas(OutOfGasError::Precompile).into())
                | (*reason == HaltReason::OutOfGas(OutOfGasError::InvalidOperand).into())
                | (*reason == HaltReason::OutOfGas(OutOfGasError::ReentrancySentry).into())
        } else {
            false
        }
    }

    /// Returns whether the exit code is a revert.
    pub fn is_revert(&self) -> bool {
        matches!(self, Self::Revert)
    }
}
