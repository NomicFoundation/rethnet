//! Defines the source model used to perform the stack trace decoding.
//!
//! The source model consists of the following:
//! - [`SourceFile`]s and their name, content
//!   - [`SourceLocation`]s that point inside the source files
//! - [`Contract`]s and its name, location
//!   - related contract and free [`ContractFunction`]s and their name, location
//!     and parameters
//!   - related [`CustomError`]s and their name, location and parameters
//! - the resolved [`ContractMetadata`] of the contract
//!   - related resolved [`Instruction`]s and their location

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

use alloy_dyn_abi::ErrorExt;
use edr_eth::{bytecode::opcode::OpCode, hex};
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::Serialize;
use serde_json::Value;

use crate::artifacts::{ContractAbiEntry, ImmutableReference};

/// A resolved build model from a Solidity compiler standard JSON output.
#[derive(Debug, Default)]
pub struct BuildModel {
    // TODO https://github.com/NomicFoundation/edr/issues/759
    /// Maps the contract ID to the contract.
    pub contract_id_to_contract: IndexMap<u32, Arc<RwLock<Contract>>>,
    /// Maps the file ID to the source file.
    pub file_id_to_source_file: Arc<BuildModelSources>,
}

// TODO https://github.com/NomicFoundation/edr/issues/759
/// Type alias for the source file mapping used by [`BuildModel`].
pub type BuildModelSources = HashMap<u32, Arc<RwLock<SourceFile>>>;

/// A source file.
#[derive(Debug)]
pub struct SourceFile {
    // Referenced because it can be later updated by outside code
    functions: Vec<Arc<ContractFunction>>,

    /// The name of the source file.
    pub source_name: String,
    /// The content of the source file.
    pub content: String,
}

impl SourceFile {
    /// Creates a new [`SourceFile`] with the provided name and content.
    pub fn new(source_name: String, content: String) -> SourceFile {
        SourceFile {
            functions: Vec::new(),

            content,
            source_name,
        }
    }

    /// Adds a [`ContractFunction`] to the source file.
    /// # Note
    /// Should only be called when resolving the source model.
    pub fn add_function(&mut self, contract_function: Arc<ContractFunction>) {
        self.functions.push(contract_function);
    }

    /// Returns the [`ContractFunction`] that contains the provided
    /// [`SourceLocation`].
    pub fn get_containing_function(
        &self,
        location: &SourceLocation,
    ) -> Option<&Arc<ContractFunction>> {
        self.functions
            .iter()
            .find(|func| func.location.contains(location))
    }
}

#[derive(Clone, Debug)]
/// A source location that is tied to a source file.
///
/// # Note
/// This is a weak reference to the source file. If the source file is dropped,
/// we can no longer access it through this reference.
pub struct SourceLocation {
    /// Cached 1-based line number of the source location.
    line: OnceLock<u32>,
    /// A weak reference to the source files mapping.
    ///
    /// Used to access the source file when needed.
    pub(crate) sources: Weak<BuildModelSources>,
    /// The file ID of the source file.
    pub file_id: u32,
    /// Byte offset of the source location.
    pub offset: u32,
    /// Byte length of the source location.
    pub length: u32,
}

impl PartialEq for SourceLocation {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.sources, &other.sources)
            && self.file_id == other.file_id
            && self.offset == other.offset
            && self.length == other.length
    }
}

impl SourceLocation {
    /// Creates a new [`SourceLocation`] with the provided file ID, offset, and
    /// length.
    pub fn new(
        sources: Arc<BuildModelSources>,
        file_id: u32,
        offset: u32,
        length: u32,
    ) -> SourceLocation {
        SourceLocation {
            line: OnceLock::new(),
            // We need to break the cycle between SourceLocation and SourceFile
            // (via ContractFunction); the Bytecode struct is owning the build
            // model sources, so we should always be alive.
            sources: Arc::downgrade(&sources),
            file_id,
            offset,
            length,
        }
    }

    /// Returns the file that contains the given source location.
    /// # Panics
    /// This function panics if the source location is dangling, i.e. source
    /// files mapping has been dropped (currently only owned by the
    /// [`ContractMetadata`]).
    pub fn file(&self) -> Arc<RwLock<SourceFile>> {
        match self.sources.upgrade() {
            Some(ref sources) => sources.get(&self.file_id).unwrap().clone(),
            None => panic!("dangling SourceLocation; did you drop the owning Bytecode?"),
        }
    }

    /// Returns the 1-based line number of the source location.
    pub fn get_starting_line_number(&self) -> u32 {
        if let Some(line) = self.line.get() {
            return *line;
        }

        let file = self.file();
        let contents = &file.read().content;

        *self.line.get_or_init(move || {
            let mut line = 1;

            for c in contents.chars().take(self.offset as usize) {
                if c == '\n' {
                    line += 1;
                }
            }

            line
        })
    }

    /// Returns the [`ContractFunction`] that contains the source location.
    pub fn get_containing_function(&self) -> Option<Arc<ContractFunction>> {
        let file = self.file();
        let file = file.read();
        file.get_containing_function(self).cloned()
    }

    /// Returns whether the source location is contained within the other source
    /// location.
    pub fn contains(&self, other: &SourceLocation) -> bool {
        if !Weak::ptr_eq(&self.sources, &other.sources) || self.file_id != other.file_id {
            return false;
        }

        if other.offset < self.offset {
            return false;
        }

        other.offset + other.length <= self.offset + self.length
    }
}

/// The type of a contract function.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ContractFunctionType {
    Constructor,
    Function,
    Fallback,
    Receive,
    Getter,
    Modifier,
    FreeFunction,
}

/// The visibility of a contract function.
#[allow(missing_docs)]
#[derive(Debug, PartialEq)]
pub enum ContractFunctionVisibility {
    Private,
    Internal,
    Public,
    External,
}

/// A contract function.
#[derive(Debug)]
pub struct ContractFunction {
    /// The name of the contract function.
    pub name: String,
    /// The type of the contract function.
    pub r#type: ContractFunctionType,
    /// The source location of the contract function.
    pub location: Arc<SourceLocation>,
    /// The name of the contract that contains the contract function.
    pub contract_name: Option<String>,
    /// The visibility of the contract function.
    pub visibility: Option<ContractFunctionVisibility>,
    /// Whether the contract function is payable.
    pub is_payable: Option<bool>,
    /// The selector of the contract function.
    /// May be fixed up by [`Contract::correct_selector`].
    pub selector: RwLock<Option<Vec<u8>>>,
    /// JSON ABI parameter types of the contract function.
    pub param_types: Option<Vec<Value>>,
}

impl<'a> TryFrom<&'a ContractFunction> for alloy_json_abi::Function {
    type Error = serde_json::Error;

    fn try_from(value: &'a ContractFunction) -> Result<Self, Self::Error> {
        let inputs = value
            .param_types
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(alloy_json_abi::Function {
            name: value.name.clone(),
            inputs,
            outputs: vec![],
            state_mutability: match value.is_payable {
                Some(true) => alloy_json_abi::StateMutability::Payable,
                _ => alloy_json_abi::StateMutability::default(),
            },
        })
    }
}

#[derive(Debug)]
/// A custom error.
pub struct CustomError {
    /// The 4-byte selector of the custom error.
    pub selector: [u8; 4],
    /// The name of the custom error.
    pub name: String,
    /// JSON ABI parameter types of the custom error.
    pub param_types: Vec<Value>,

    def: alloy_json_abi::Error,
}

impl CustomError {
    /// Creates a new [`CustomError`] from the provided [`ContractAbiEntry`].
    pub fn from_abi(entry: ContractAbiEntry) -> Result<CustomError, Box<str>> {
        // FIXME(#636): This is wasteful; to fix that we'd have to implement
        // tighter deserialization for the contract ABI entries.
        let json = serde_json::to_value(&entry).expect("ContractAbiEntry to be round-trippable");

        let selector = crate::utils::json_abi_error_selector(&json)?;

        Ok(CustomError {
            selector,
            name: entry.name.expect("ABI errors to always have names"),
            param_types: entry.inputs.unwrap_or_default(),
            def: serde_json::from_value(json).map_err(|e| e.to_string().into_boxed_str())?,
        })
    }

    /// Decodes the error data (*with* selector).
    pub fn decode_error_data(
        &self,
        data: &[u8],
    ) -> alloy_dyn_abi::Result<alloy_dyn_abi::DecodedError> {
        self.def.decode_error(data)
    }
}

/// A decoded EVM instruction.
#[derive(Clone, Debug)]
pub struct Instruction {
    /// The program counter (PC) of the instruction in a bytecode.
    pub pc: u32,
    /// The opcode of the instruction.
    pub opcode: OpCode,
    /// The jump type of the instruction, if any.
    pub jump_type: JumpType,
    /// The push data of the instruction, if it's a `PUSHx` instruction.
    pub push_data: Option<Vec<u8>>,
    /// The source location of the instruction, if any.
    pub location: Option<Arc<SourceLocation>>,
}

/// The type of a jump.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::Display)]
pub enum JumpType {
    /// The instruction is not a jump.
    NotJump,
    /// The instruction is a jump into a function.
    IntoFunction,
    /// The instruction is a jump out of a function.
    OutofFunction,
    /// The instruction is an internal jump, e.g. a loop.
    InternalJump,
}

/// A [`ContractMetadata`] error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ContractMetadataError {
    /// The instruction was not found at the provided program counter (PC).
    #[error("Instruction not found at PC {pc}")]
    InstructionNotFound {
        /// The program counter (PC) of the instruction.
        pc: u32,
    },
}

/// A resolved bytecode.
#[derive(Debug)]
pub struct ContractMetadata {
    pc_to_instruction: HashMap<u32, Instruction>,

    // This owns the source files transitively used by the source locations
    // in the Instruction structs.
    _sources: Arc<BuildModelSources>,
    // TODO https://github.com/NomicFoundation/edr/issues/759
    /// Contract that the bytecode belongs to.
    pub contract: Arc<RwLock<Contract>>,
    /// Whether the bytecode is a deployment bytecode.
    pub is_deployment: bool,
    /// Normalized code of the bytecode, i.e. replaced with zeroes for the
    /// library addresses.
    pub normalized_code: Vec<u8>,
    /// Positions in the bytecode of the library addresses.
    pub library_address_positions: Vec<u32>,
    /// Positions in the bytecode of the immutable references.
    pub immutable_references: Vec<ImmutableReference>,
    /// Solidity compiler version used to compile the bytecode.
    pub compiler_version: String,
}

impl ContractMetadata {
    /// Creates a new [`ContractMetadata`] with the provided arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: Arc<BuildModelSources>,
        contract: Arc<RwLock<Contract>>,
        is_deployment: bool,
        normalized_code: Vec<u8>,
        instructions: Vec<Instruction>,
        library_address_positions: Vec<u32>,
        immutable_references: Vec<ImmutableReference>,
        compiler_version: String,
    ) -> ContractMetadata {
        let mut pc_to_instruction = HashMap::new();
        for inst in instructions {
            pc_to_instruction.insert(inst.pc, inst);
        }

        ContractMetadata {
            pc_to_instruction,
            _sources: sources,
            contract,
            is_deployment,
            normalized_code,
            library_address_positions,
            immutable_references,
            compiler_version,
        }
    }

    /// Returns the [`Instruction`] at the provided program counter (PC).
    pub fn get_instruction(&self, pc: u32) -> Result<&Instruction, ContractMetadataError> {
        self.pc_to_instruction
            .get(&pc)
            .ok_or(ContractMetadataError::InstructionNotFound { pc })
    }

    /// Returns the [`Instruction`] at the provided program counter (PC). The
    /// error type is `anyhow::Error` which can be converted to
    /// `napi::Error` automatically. Usage of this method is deprecated and
    /// call sites in `edr_napi` will be removed.
    #[deprecated = "Use `get_instruction` instead"]
    pub fn get_instruction_napi(&self, pc: u32) -> anyhow::Result<&Instruction> {
        self.get_instruction(pc).map_err(anyhow::Error::from)
    }

    /// Whether the bytecode has an instruction at the provided program counter
    /// (PC).
    pub fn has_instruction(&self, pc: u32) -> bool {
        self.pc_to_instruction.contains_key(&pc)
    }
}

/// The kind of a contract.
#[derive(Debug, PartialEq, strum::EnumString)]
#[strum(serialize_all = "camelCase")]
pub enum ContractKind {
    /// A contract.
    Contract,
    /// A library.
    Library,
}

/// A resolved contract.
#[derive(Debug)]
pub struct Contract {
    /// Custom errors defined in the contract.
    pub custom_errors: Vec<CustomError>,
    /// The constructor function of the contract.
    pub constructor: Option<Arc<ContractFunction>>,
    /// The fallback function of the contract.
    pub fallback: Option<Arc<ContractFunction>>,
    /// The receive function of the contract.
    pub receive: Option<Arc<ContractFunction>>,

    local_functions: Vec<Arc<ContractFunction>>,
    selector_hex_to_function: HashMap<String, Arc<ContractFunction>>,

    /// The contract's name.
    pub name: String,
    /// The contract's kind, i.e. contract or library.
    pub r#type: ContractKind,
    /// The source location of the contract.
    pub location: Arc<SourceLocation>,
}

impl Contract {
    /// Creates a new [`Contract`] with the provided arguments.
    pub fn new(
        name: String,
        contract_type: ContractKind,
        location: Arc<SourceLocation>,
    ) -> Contract {
        Contract {
            custom_errors: Vec::new(),
            constructor: None,
            fallback: None,
            receive: None,
            local_functions: Vec::new(),
            selector_hex_to_function: HashMap::new(),
            name,
            r#type: contract_type,
            location,
        }
    }

    /// Adds a local function to the contract.
    /// # Note
    /// Should only be called when resolving the source model.
    pub fn add_local_function(&mut self, func: Arc<ContractFunction>) {
        if matches!(
            func.visibility,
            Some(ContractFunctionVisibility::Public | ContractFunctionVisibility::External)
        ) {
            match func.r#type {
                ContractFunctionType::Function | ContractFunctionType::Getter => {
                    let selector = func.selector.read();
                    // The original code unwrapped here
                    let selector = selector.as_ref().unwrap();
                    let selector = hex::encode(selector);

                    self.selector_hex_to_function.insert(selector, func.clone());
                }
                ContractFunctionType::Constructor => {
                    self.constructor = Some(func.clone());
                }
                ContractFunctionType::Fallback => {
                    self.fallback = Some(func.clone());
                }
                ContractFunctionType::Receive => {
                    self.receive = Some(func.clone());
                }
                _ => {}
            }
        }

        self.local_functions.push(func);
    }

    /// Adds a custom error to the contract.
    /// # Note
    /// Should only be called when resolving the source model.
    pub fn add_custom_error(&mut self, value: CustomError) {
        self.custom_errors.push(value);
    }

    /// Adds the next linearized base contract to the contract, possibly
    /// overwriting the functions of the contract.
    /// # Note
    /// Should only be called when resolving the source model.
    pub fn add_next_linearized_base_contract(&mut self, base_contract: &Contract) {
        if self.fallback.is_none() && base_contract.fallback.is_some() {
            self.fallback.clone_from(&base_contract.fallback);
        }
        if self.receive.is_none() && base_contract.receive.is_some() {
            self.receive.clone_from(&base_contract.receive);
        }

        for base_contract_function in &base_contract.local_functions {
            let base_contract_function_clone = base_contract_function.clone();

            if base_contract_function.r#type != ContractFunctionType::Getter
                && base_contract_function.r#type != ContractFunctionType::Function
            {
                continue;
            }

            if base_contract_function.visibility != Some(ContractFunctionVisibility::Public)
                && base_contract_function.visibility != Some(ContractFunctionVisibility::External)
            {
                continue;
            }

            let selector = base_contract_function
                .selector
                .read()
                .clone()
                .expect("selector exists");
            let selector_hex = hex::encode(&*selector);

            self.selector_hex_to_function
                .entry(selector_hex)
                .or_insert(base_contract_function_clone);
        }
    }

    /// Looks up the local [`ContractFunction`] with the provided selector.
    pub fn get_function_from_selector(&self, selector: &[u8]) -> Option<&Arc<ContractFunction>> {
        let selector_hex = hex::encode(selector);

        self.selector_hex_to_function.get(&selector_hex)
    }

    /// We compute selectors manually, which is particularly hard. We do this
    /// because we need to map selectors to AST nodes, and it seems easier to
    /// start from the AST node. This is surprisingly super hard: things
    /// like inherited enums, structs and `ABIv2` complicate it.
    ///
    /// As we know that that can fail, we run a heuristic that tries to correct
    /// incorrect selectors. What it does is checking the
    /// `evm.methodIdentifiers` compiler output, and detect missing
    /// selectors. Then we take those and find contract functions with the
    /// same name. If there are multiple of those we can't do anything. If
    /// there is a single one, it must have an incorrect selector, so we
    /// update it with the `evm.methodIdentifiers`'s value.
    pub fn correct_selector(&mut self, function_name: String, selector: Vec<u8>) -> bool {
        let functions = self
            .selector_hex_to_function
            .values()
            .filter(|cf| cf.name == function_name)
            .cloned()
            .collect::<Vec<_>>();

        let function_to_correct = match functions.split_first() {
            Some((function_to_correct, [])) => function_to_correct,
            _ => return false,
        };

        {
            let mut selector_to_be_corrected = function_to_correct.selector.write();
            if let Some(selector) = &*selector_to_be_corrected {
                let selector_hex = hex::encode(selector);
                self.selector_hex_to_function.remove(&selector_hex);
            }

            *selector_to_be_corrected = Some(selector.clone());
        }

        let selector_hex = hex::encode(&*selector);
        self.selector_hex_to_function
            .insert(selector_hex, function_to_correct.clone());

        true
    }
}
