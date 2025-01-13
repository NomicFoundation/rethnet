//!  This file includes Solidity tracing heuristics for solc starting with
//! version  0.6.9.
//!
//!  This solc version introduced a significant change to how sourcemaps are
//!  handled for inline yul/internal functions. These were mapped to the
//!  unmapped/-1 file before, which lead to many unmapped reverts. Now, they are
//!  mapped to the part of the Solidity source that lead to their inlining.
//!
//!  This change is a very positive change, as errors would point to the correct
//!  line by default. The only problem is that we used to rely very heavily on
//!  unmapped reverts to decide when our error detection heuristics were to be
//!  run. In fact, these heuristics were first introduced because of unmapped
//!  reverts.
//!
//!  Instead of synthetically completing stack traces when unmapped reverts
//! occur, we now start from complete stack traces and adjust them if we can
//! provide more meaningful errors.

use edr_eth::{bytecode::opcode::OpCode, spec::HaltReasonTrait};
use semver::Version;

use crate::{
    build_model::ContractMetadataError,
    nested_trace::{CreateOrCallMessageRef, EvmStep, NestedTraceStep},
    solidity_stack_trace::StackTraceEntry,
};

const FIRST_SOLC_VERSION_WITH_MAPPED_SMALL_INTERNAL_FUNCTIONS: Version = Version::new(0, 6, 9);

#[derive(Debug, thiserror::Error)]
pub enum HeuristicsError {
    #[error(transparent)]
    BytecodeError(#[from] ContractMetadataError),
    #[error("Missing contract")]
    MissingContract,
}

pub fn stack_trace_may_require_adjustments<HaltReasonT: HaltReasonTrait>(
    stacktrace: &[StackTraceEntry],
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, HeuristicsError> {
    let contract_meta = decoded_trace
        .contract_meta()
        .ok_or(HeuristicsError::MissingContract)?;

    let Some(last_frame) = stacktrace.last() else {
        return Ok(false);
    };

    if let StackTraceEntry::RevertError {
        is_invalid_opcode_error,
        return_data,
        ..
    } = last_frame
    {
        let result = !is_invalid_opcode_error
            && return_data.is_empty()
            && Version::parse(&contract_meta.compiler_version)
                .map(|version| version >= FIRST_SOLC_VERSION_WITH_MAPPED_SMALL_INTERNAL_FUNCTIONS)
                .unwrap_or(false);
        return Ok(result);
    }

    Ok(false)
}

pub fn adjust_stack_trace<HaltReasonT: HaltReasonTrait>(
    mut stacktrace: Vec<StackTraceEntry>,
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<Vec<StackTraceEntry>, HeuristicsError> {
    let Some(StackTraceEntry::RevertError {
        source_reference, ..
    }) = stacktrace.last()
    else {
        unreachable!("This should be only used immediately after we check with `stack_trace_may_require_adjustments` that the last frame is a revert frame");
    };

    // Replace the last revert frame with an adjusted frame if needed
    if is_non_contract_account_called_error(decoded_trace)? {
        let last_revert_frame_source_reference = source_reference.clone();
        stacktrace.pop();
        stacktrace.push(StackTraceEntry::NoncontractAccountCalledError {
            source_reference: last_revert_frame_source_reference,
        });
        return Ok(stacktrace);
    }

    if is_constructor_invalid_params_error(decoded_trace)? {
        let last_revert_frame_source_reference = source_reference.clone();
        stacktrace.pop();
        stacktrace.push(StackTraceEntry::InvalidParamsError {
            source_reference: last_revert_frame_source_reference,
        });
        return Ok(stacktrace);
    }

    if is_call_invalid_params_error(decoded_trace)? {
        let last_revert_frame_source_reference = source_reference.clone();
        stacktrace.pop();
        stacktrace.push(StackTraceEntry::InvalidParamsError {
            source_reference: last_revert_frame_source_reference,
        });

        return Ok(stacktrace);
    }

    Ok(stacktrace)
}

fn is_non_contract_account_called_error<HaltReasonT: HaltReasonTrait>(
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, HeuristicsError> {
    match_opcodes(
        decoded_trace,
        -9,
        &[
            OpCode::EXTCODESIZE,
            OpCode::ISZERO,
            OpCode::DUP1,
            OpCode::ISZERO,
        ],
    )
}

fn is_constructor_invalid_params_error<HaltReasonT: HaltReasonTrait>(
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, HeuristicsError> {
    Ok(match_opcodes(decoded_trace, -20, &[OpCode::CODESIZE])?
        && match_opcodes(decoded_trace, -15, &[OpCode::CODECOPY])?
        && match_opcodes(decoded_trace, -7, &[OpCode::LT, OpCode::ISZERO])?)
}

fn is_call_invalid_params_error<HaltReasonT: HaltReasonTrait>(
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
) -> Result<bool, HeuristicsError> {
    Ok(match_opcodes(decoded_trace, -11, &[OpCode::CALLDATASIZE])?
        && match_opcodes(decoded_trace, -7, &[OpCode::LT, OpCode::ISZERO])?)
}

fn match_opcodes<HaltReasonT: HaltReasonTrait>(
    decoded_trace: CreateOrCallMessageRef<'_, HaltReasonT>,
    first_step_index: i32,
    opcodes: &[OpCode],
) -> Result<bool, HeuristicsError> {
    let contract_meta = decoded_trace
        .contract_meta()
        .ok_or(HeuristicsError::MissingContract)?;
    let steps = decoded_trace.steps();

    // If the index is negative, we start from the end of the trace,
    // just like in the original JS code
    let mut index = match first_step_index {
        0.. => first_step_index as usize,
        ..=-1 if first_step_index.abs() < steps.len() as i32 => {
            (steps.len() as i32 + first_step_index) as usize
        }
        // Out of bounds
        _ => return Ok(false),
    };

    for opcode in opcodes {
        let Some(NestedTraceStep::Evm(EvmStep { pc })) = steps.get(index) else {
            return Ok(false);
        };

        let instruction = contract_meta.get_instruction(*pc)?;

        if instruction.opcode != *opcode {
            return Ok(false);
        }

        index += 1;
    }

    Ok(true)
}
