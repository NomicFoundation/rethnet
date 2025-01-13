//! Naive Rust port of the `MessageTrace` et al. from Hardhat.

use std::sync::Arc;

use derive_where::derive_where;
use edr_eth::{spec::HaltReasonTrait, Address, Bytes, U256};

use crate::{build_model::ContractMetadata, exit_code::ExitCode};

/// An EVM trace where the steps are nested according to the call stack.
#[derive(Clone, Debug)]
pub enum NestedTrace<HaltReasonT: HaltReasonTrait> {
    /// Represents a create trace.
    Create(CreateMessage<HaltReasonT>),
    /// Represents a call trace.
    Call(CallMessage<HaltReasonT>),
    /// Represents a precompile trace.
    Precompile(PrecompileMessage<HaltReasonT>),
}

impl<HaltReasonT: HaltReasonTrait> NestedTrace<HaltReasonT> {
    /// Returns the exit code of the trace.
    pub fn exit_code(&self) -> &ExitCode<HaltReasonT> {
        match self {
            Self::Create(create) => &create.exit,
            Self::Call(call) => &call.exit,
            Self::Precompile(precompile) => &precompile.exit,
        }
    }
}

/// Represents a precompile message.
#[derive(Clone, Debug)]
pub struct PrecompileMessage<HaltReasonT: HaltReasonTrait> {
    /// Precompile number.
    pub precompile: u32,
    /// Calldata buffer
    pub calldata: Bytes,
    /// Value of the message.
    pub value: U256,
    /// Return data buffer.
    pub return_data: Bytes,
    /// EVM exit code.
    pub exit: ExitCode<HaltReasonT>,
    /// How much gas was used.
    pub gas_used: u64,
    /// Depth of the message.
    pub depth: usize,
}

/// Represents a create message.
#[derive(Clone, Debug)]
pub struct CreateMessage<HaltReasonT: HaltReasonTrait> {
    // The following is just an optimization: When processing this traces it's useful to know ahead
    // of time how many subtraces there are.
    /// Number of subtraces. Used to speed up the processing of the traces in
    /// JS.
    pub number_of_subtraces: u32,
    /// Children messages.
    pub steps: Vec<NestedTraceStep<HaltReasonT>>,
    /// Resolved metadata of the contract that is being executed.
    pub contract_meta: Option<Arc<ContractMetadata>>,
    /// Address of the deployed contract.
    pub deployed_contract: Option<Bytes>,
    /// Code of the contract that is being executed.
    pub code: Bytes,
    /// Value of the message.
    pub value: U256,
    /// Return data buffer.
    pub return_data: Bytes,
    /// EVM exit code.
    pub exit: ExitCode<HaltReasonT>,
    /// How much gas was used.
    pub gas_used: u64,
    /// Depth of the message.
    pub depth: usize,
}

/// Represents a call message with contract metadata.
#[derive(Clone, Debug)]
pub struct CallMessage<HaltReasonT: HaltReasonTrait> {
    // The following is just an optimization: When processing this traces it's useful to know ahead
    // of time how many subtraces there are.
    /// Number of subtraces. Used to speed up the processing of the traces in
    /// JS.
    pub number_of_subtraces: u32,
    /// Children messages.
    pub steps: Vec<NestedTraceStep<HaltReasonT>>,
    /// Resolved metadata of the contract that is being executed.
    pub contract_meta: Option<Arc<ContractMetadata>>,
    /// Calldata buffer
    pub calldata: Bytes,
    /// Address of the contract that is being executed.
    pub address: Address,
    /// Address of the code that is being executed.
    pub code_address: Address,
    /// Code of the contract that is being executed.
    pub code: Bytes,
    /// Value of the message.
    pub value: U256,
    /// Return data buffer.
    pub return_data: Bytes,
    /// EVM exit code.
    pub exit: ExitCode<HaltReasonT>,
    /// How much gas was used.
    pub gas_used: u64,
    /// Depth of the message.
    pub depth: usize,
}

/// Represents a create or call message.
#[derive(Clone, Debug)]
pub enum CreateOrCallMessage<HaltReasonT: HaltReasonTrait> {
    /// Represents a create message.
    Create(CreateMessage<HaltReasonT>),
    /// Represents a call message.
    Call(CallMessage<HaltReasonT>),
}

impl<HaltReasonT: HaltReasonTrait> From<CreateMessage<HaltReasonT>>
    for CreateOrCallMessage<HaltReasonT>
{
    fn from(value: CreateMessage<HaltReasonT>) -> Self {
        CreateOrCallMessage::Create(value)
    }
}

impl<HaltReasonT: HaltReasonTrait> From<CallMessage<HaltReasonT>>
    for CreateOrCallMessage<HaltReasonT>
{
    fn from(value: CallMessage<HaltReasonT>) -> Self {
        CreateOrCallMessage::Call(value)
    }
}

/// Represents a create or call message.
#[derive(Debug)]
#[derive_where(Clone, Copy)]
pub(crate) enum CreateOrCallMessageRef<'a, HaltReasonT: HaltReasonTrait> {
    /// Represents a create message.
    Create(&'a CreateMessage<HaltReasonT>),
    /// Represents a call message.
    Call(&'a CallMessage<HaltReasonT>),
}

impl<'a, HaltReasonT: HaltReasonTrait> CreateOrCallMessageRef<'a, HaltReasonT> {
    pub fn contract_meta(&self) -> Option<Arc<ContractMetadata>> {
        match self {
            CreateOrCallMessageRef::Create(create) => create.contract_meta.as_ref().map(Arc::clone),
            CreateOrCallMessageRef::Call(call) => call.contract_meta.as_ref().map(Arc::clone),
        }
    }

    pub fn exit_code(&self) -> &ExitCode<HaltReasonT> {
        match self {
            CreateOrCallMessageRef::Create(create) => &create.exit,
            CreateOrCallMessageRef::Call(call) => &call.exit,
        }
    }

    pub fn number_of_subtraces(&self) -> u32 {
        match self {
            CreateOrCallMessageRef::Create(create) => create.number_of_subtraces,
            CreateOrCallMessageRef::Call(call) => call.number_of_subtraces,
        }
    }

    pub fn return_data(&self) -> &Bytes {
        match self {
            CreateOrCallMessageRef::Create(create) => &create.return_data,
            CreateOrCallMessageRef::Call(call) => &call.return_data,
        }
    }

    pub fn steps(&self) -> &'a [NestedTraceStep<HaltReasonT>] {
        match self {
            CreateOrCallMessageRef::Create(create) => create.steps.as_slice(),
            CreateOrCallMessageRef::Call(call) => call.steps.as_slice(),
        }
    }
}

impl<'a, HaltReasonT: HaltReasonTrait> From<&'a CreateOrCallMessage<HaltReasonT>>
    for CreateOrCallMessageRef<'a, HaltReasonT>
{
    fn from(value: &'a CreateOrCallMessage<HaltReasonT>) -> Self {
        match value {
            CreateOrCallMessage::Create(create) => CreateOrCallMessageRef::Create(create),
            CreateOrCallMessage::Call(call) => CreateOrCallMessageRef::Call(call),
        }
    }
}

impl<'a, HaltReasonT: HaltReasonTrait> From<&'a CreateMessage<HaltReasonT>>
    for CreateOrCallMessageRef<'a, HaltReasonT>
{
    fn from(value: &'a CreateMessage<HaltReasonT>) -> Self {
        CreateOrCallMessageRef::Create(value)
    }
}

impl<'a, HaltReasonT: HaltReasonTrait> From<&'a CallMessage<HaltReasonT>>
    for CreateOrCallMessageRef<'a, HaltReasonT>
{
    fn from(value: &'a CallMessage<HaltReasonT>) -> Self {
        CreateOrCallMessageRef::Call(value)
    }
}

/// Represents a nested trace step with contract metadata.
#[derive(Clone, Debug)]
pub enum NestedTraceStep<HaltReasonT: HaltReasonTrait> {
    /// Represents a create message.
    Create(CreateMessage<HaltReasonT>),
    /// Represents a call message.
    Call(CallMessage<HaltReasonT>),
    /// Represents a precompile message.
    Precompile(PrecompileMessage<HaltReasonT>),
    /// Minimal EVM step that contains only PC (program counter).
    Evm(EvmStep),
}

/// Minimal EVM step that contains only PC (program counter).
#[derive(Clone, Debug)]
pub struct EvmStep {
    /// Program counter
    pub pc: u32,
}
