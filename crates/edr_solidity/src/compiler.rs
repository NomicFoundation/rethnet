//! Processes the Solidity compiler standard JSON[^1] input and output AST and
//! creates the source model used to perform the stack trace decoding.
//!
//! [^1]: See <https://docs.soliditylang.org/en/latest/using-the-compiler.html#compiler-input-and-output-json-description>.
use std::{collections::HashMap, str::FromStr, sync::Arc};

use anyhow::{self, Context as _};
use edr_eth::{hex, keccak256};
use indexmap::IndexMap;
use parking_lot::RwLock;

use crate::{
    artifacts::{CompilerInput, CompilerOutput, CompilerOutputBytecode, ContractAbiEntry},
    build_model::{
        BuildModel, BuildModelSources, Contract, ContractFunction, ContractFunctionType,
        ContractFunctionVisibility, ContractKind, ContractMetadata, CustomError, SourceFile,
        SourceLocation,
    },
    library_utils::{get_library_address_positions, normalize_compiler_output_bytecode},
    source_map::decode_instructions,
};

/// For the Solidity compiler version and its standard JSON input and
/// output, creates the source model, decodes the bytecode with source
/// mapping and links them to the source files.
///
/// Returns the decoded bytecodes that reference the resolved source model.
pub fn create_models_and_decode_bytecodes(
    solc_version: String,
    compiler_input: &CompilerInput,
    compiler_output: &CompilerOutput,
) -> anyhow::Result<Vec<ContractMetadata>> {
    let build_model = create_sources_model_from_ast(compiler_output, compiler_input)?;
    let build_model = Arc::new(build_model);

    let bytecodes = decode_bytecodes(solc_version, compiler_output, &build_model)?;

    correct_selectors(&bytecodes, compiler_output)?;

    Ok(bytecodes)
}

fn create_sources_model_from_ast(
    compiler_output: &CompilerOutput,
    compiler_input: &CompilerInput,
) -> anyhow::Result<BuildModel> {
    // First, collect and store all the files to be able to resolve the source
    // locations
    let sources: Arc<HashMap<_, _>> = Arc::new(
        compiler_output
            .sources
            .iter()
            .map(|(source_name, source)| {
                let file = SourceFile::new(
                    source_name.clone(),
                    compiler_input.sources[source_name].content.clone(),
                );
                let file = Arc::new(RwLock::new(file));
                (source.id, file.clone())
            })
            .collect(),
    );
    let mut contract_id_to_linearized_base_contract_ids = HashMap::new();

    // Secondly, collect all the contracts and fill the source file/contracts with
    // processed functions
    let mut contract_id_to_contract = IndexMap::new();
    for (source_name, source) in &compiler_output.sources {
        let file = &sources[&source.id];

        for node in source.ast["nodes"].as_array().unwrap() {
            match node["nodeType"].as_str().unwrap() {
                "ContractDefinition" => {
                    let Some(contract_type) = node["contractKind"]
                        .as_str()
                        .and_then(|k| ContractKind::from_str(k).ok())
                    else {
                        continue;
                    };

                    let contract_abi =
                        compiler_output
                            .contracts
                            .get(source_name)
                            .and_then(|contracts| {
                                contracts
                                    .get(node["name"].as_str().unwrap())
                                    .map(|contract| &contract.abi)
                            });

                    let (contract_id, contract) = process_contract_ast_node(
                        file,
                        node,
                        contract_type,
                        &sources,
                        &mut contract_id_to_linearized_base_contract_ids,
                        contract_abi.map(Vec::as_slice),
                    )?;

                    contract_id_to_contract.insert(contract_id, contract);
                }
                // top-level functions
                "FunctionDefinition" => {
                    process_function_definition_ast_node(node, &sources, None, file, None)?;
                }
                _ => {}
            }
        }
    }

    apply_contracts_inheritance(
        &contract_id_to_contract,
        &contract_id_to_linearized_base_contract_ids,
    );

    Ok(BuildModel {
        file_id_to_source_file: sources,
        contract_id_to_contract,
    })
}

fn apply_contracts_inheritance(
    contract_id_to_contract: &IndexMap<u32, Arc<RwLock<Contract>>>,
    contract_id_to_linearized_base_contract_ids: &HashMap<u32, Vec<u32>>,
) {
    for (cid, contract) in contract_id_to_contract {
        let mut contract = contract.write();

        let inheritance_ids = &contract_id_to_linearized_base_contract_ids[cid];

        for base_id in inheritance_ids {
            let base_contract = contract_id_to_contract.get(base_id);

            let base_contract = match base_contract {
                Some(base_contract) => base_contract,
                // This list includes interface, which we don't model
                None => continue,
            };

            if cid != base_id {
                let base_contract = &base_contract.read();
                contract.add_next_linearized_base_contract(base_contract);
            }
        }
    }
}

fn process_contract_ast_node(
    file: &RwLock<SourceFile>,
    contract_node: &serde_json::Value,
    contract_type: ContractKind,
    sources: &Arc<BuildModelSources>,
    contract_id_to_linearized_base_contract_ids: &mut HashMap<u32, Vec<u32>>,
    contract_abi: Option<&[ContractAbiEntry]>,
) -> anyhow::Result<(u32, Arc<RwLock<Contract>>)> {
    let contract_location =
        ast_src_to_source_location(contract_node["src"].as_str().unwrap(), sources)?
            .expect("The original JS code always asserts that");

    let contract = Contract::new(
        contract_node["name"].as_str().unwrap().to_string(),
        contract_type,
        contract_location,
    );
    let contract = Arc::new(RwLock::new(contract));

    let contract_id = contract_node["id"].as_u64().unwrap() as u32;

    contract_id_to_linearized_base_contract_ids.insert(
        contract_id,
        contract_node["linearizedBaseContracts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
    );

    for node in contract_node["nodes"].as_array().unwrap() {
        match node["nodeType"].as_str().unwrap() {
            "FunctionDefinition" => {
                let function_abis = contract_abi.map(|contract_abi| {
                    contract_abi
                        .iter()
                        .filter(|abi_entry| abi_entry.name.as_deref() == node["name"].as_str())
                        .collect::<Vec<_>>()
                });

                process_function_definition_ast_node(
                    node,
                    sources,
                    Some(&contract),
                    file,
                    function_abis,
                )?;
            }
            "ModifierDefinition" => {
                process_modifier_definition_ast_node(node, sources, &contract, file)?;
            }
            "VariableDeclaration" => {
                let getter_abi = contract_abi.and_then(|contract_abi| {
                    contract_abi
                        .iter()
                        .find(|abi_entry| abi_entry.name.as_deref() == node["name"].as_str())
                });

                process_variable_declaration_ast_node(node, sources, &contract, file, getter_abi)?;
            }
            _ => {}
        }
    }

    Ok((contract_id, contract))
}

fn process_function_definition_ast_node(
    node: &serde_json::Value,
    sources: &Arc<BuildModelSources>,
    contract: Option<&RwLock<Contract>>,
    file: &RwLock<SourceFile>,
    function_abis: Option<Vec<&ContractAbiEntry>>,
) -> anyhow::Result<()> {
    if node.get("implemented").and_then(serde_json::Value::as_bool) == Some(false) {
        return Ok(());
    }

    let function_type = function_definition_kind_to_function_type(node["kind"].as_str());

    let function_location = ast_src_to_source_location(node["src"].as_str().unwrap(), sources)?
        .expect("The original JS code always asserts that");

    let visibility = ast_visibility_to_visibility(node["visibility"].as_str().unwrap());

    let selector = if function_type == ContractFunctionType::Function
        && (visibility == ContractFunctionVisibility::External
            || visibility == ContractFunctionVisibility::Public)
    {
        Some(ast_function_definition_to_selector(node)?)
    } else {
        None
    };

    // function can be overloaded, match the abi by the selector
    let matching_function_abi = function_abis.as_ref().and_then(|function_abis| {
        function_abis.iter().find(|function_abi| {
            let name = match function_abi.name {
                Some(ref name) => name,
                None => return false,
            };

            let function_abi_selector = abi_method_id(
                name,
                function_abi
                    .inputs
                    .as_ref()
                    .map(|inputs| {
                        inputs
                            .iter()
                            .map(|input| input["type"].as_str().unwrap())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            );

            match (selector.as_ref(), function_abi_selector) {
                (Some(selector), function_abi_selector) if !function_abi_selector.is_empty() => {
                    selector.as_ref() == function_abi_selector
                }
                _ => false,
            }
        })
    });

    let param_types = matching_function_abi
        .as_ref()
        .and_then(|abi| abi.inputs.as_ref())
        .cloned();

    let contract_func = ContractFunction {
        name: node["name"].as_str().unwrap().to_string(),
        r#type: function_type,
        location: function_location,
        contract_name: contract.as_ref().map(|c| c.read()).map(|c| c.name.clone()),
        visibility: Some(visibility),
        is_payable: Some(node["stateMutability"].as_str().unwrap() == "payable"),
        selector: RwLock::new(selector),
        param_types,
    };
    let contract_func = Arc::new(contract_func);

    file.write().add_function(contract_func.clone());
    if let Some(contract) = contract {
        contract.write().add_local_function(contract_func);
    }

    Ok(())
}

fn process_modifier_definition_ast_node(
    node: &serde_json::Value,
    sources: &Arc<BuildModelSources>,
    contract: &RwLock<Contract>,
    file: &RwLock<SourceFile>,
) -> anyhow::Result<()> {
    let function_location = ast_src_to_source_location(node["src"].as_str().unwrap(), sources)?
        .expect("The original JS code always asserts that");

    let contract_func = ContractFunction {
        name: node["name"].as_str().unwrap().to_string(),
        r#type: ContractFunctionType::Modifier,
        location: function_location,
        contract_name: Some(contract.read().name.clone()),
        visibility: None,
        is_payable: None,
        selector: RwLock::new(None),
        param_types: None,
    };

    let contract_func = Arc::new(contract_func);

    file.write().add_function(contract_func.clone());
    contract.write().add_local_function(contract_func);

    Ok(())
}

fn process_variable_declaration_ast_node(
    node: &serde_json::Value,
    sources: &Arc<BuildModelSources>,
    contract: &RwLock<Contract>,
    file: &RwLock<SourceFile>,
    getter_abi: Option<&ContractAbiEntry>,
) -> anyhow::Result<()> {
    let visibility = ast_visibility_to_visibility(node["visibility"].as_str().unwrap());

    // Variables can't be external
    if visibility != ContractFunctionVisibility::Public {
        return Ok(());
    }

    let function_location = ast_src_to_source_location(node["src"].as_str().unwrap(), sources)?
        .expect("The original JS code always asserts that");

    let param_types = getter_abi
        .as_ref()
        .and_then(|abi| abi.inputs.as_ref())
        .cloned();

    let contract_func = ContractFunction {
        name: node["name"].as_str().unwrap().to_string(),
        r#type: ContractFunctionType::Getter,
        location: function_location,
        contract_name: Some(contract.read().name.clone()),
        visibility: Some(visibility),
        is_payable: Some(false), // Getters aren't payable
        selector: RwLock::new(Some(
            get_public_variable_selector_from_declaration_ast_node(node)?,
        )),
        param_types,
    };
    let contract_func = Arc::new(contract_func);

    file.write().add_function(contract_func.clone());
    contract.write().add_local_function(contract_func);

    Ok(())
}

fn get_public_variable_selector_from_declaration_ast_node(
    variable_declaration: &serde_json::Value,
) -> anyhow::Result<Vec<u8>> {
    if let Some(function_selector) = variable_declaration["functionSelector"].as_str() {
        return hex::decode(function_selector)
            .with_context(|| format!("Failed to decode hex: {function_selector:?}"));
    }

    // NOTE: It seems we don't have tests that exercise missing functionSelector
    // in the variable declaration
    let mut param_types = Vec::new();

    // VariableDeclaration nodes for function parameters or state variables will
    // always have their typeName fields defined.
    let mut next_type = &variable_declaration["typeName"];
    loop {
        if next_type["nodeType"] == "Mapping" {
            let canonical_type =
                canonical_abi_type_for_elementary_or_user_defined_types(&next_type["keyType"])
                    .expect("Original code asserted that");

            param_types.push(canonical_type);

            next_type = &next_type["valueType"];
        } else {
            if next_type["nodeType"] == "ArrayTypeName" {
                param_types.push("uint256".to_string());
            }

            break;
        }
    }

    let method_id = abi_method_id(variable_declaration["name"].as_str().unwrap(), param_types);

    Ok(method_id)
}

fn ast_function_definition_to_selector(
    function_definition: &serde_json::Value,
) -> anyhow::Result<Vec<u8>> {
    if let Some(function_selector) = function_definition["functionSelector"].as_str() {
        return hex::decode(function_selector)
            .with_context(|| format!("Failed to decode hex: {function_selector:?}"));
    }

    let mut param_types = Vec::new();

    for param in function_definition["parameters"]["parameters"]
        .as_array()
        .unwrap()
    {
        if is_contract_type(param) {
            param_types.push("address".to_string());
            continue;
        }

        // TODO: implement ABIv2 structs parsing
        // This might mean we need to parse struct definitions before
        // resolving types and trying to calculate function selectors.
        // if is_struct_type(param) {
        //   param_types.push(something);
        //   continue;
        // }

        if is_enum_type(param) {
            // TODO: If the enum has >= 256 elements this will fail. It should be a uint16.
            // This is  complicated, as enums can be inherited. Fortunately, if
            // multiple parent contracts  define the same enum, solc fails to
            // compile.
            param_types.push("uint8".to_string());
            continue;
        }

        let typename = &param["typeName"];
        let node_type = param
            .pointer("/typeName/nodeType")
            .and_then(serde_json::Value::as_str);
        if matches!(
            node_type,
            Some("ArrayTypeName" | "FunctionTypeName" | "Mapping")
        ) {
            param_types.push(
                typename["typeDescriptions"]["typeString"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            );
            continue;
        }

        param_types.push(to_canonical_abi_type(typename["name"].as_str().unwrap()));
    }

    Ok(abi_method_id(
        function_definition["name"].as_str().unwrap(),
        param_types,
    ))
}

fn canonical_abi_type_for_elementary_or_user_defined_types(
    key_type: &serde_json::Value,
) -> Option<String> {
    if is_elementary_type(key_type) {
        return Some(to_canonical_abi_type(key_type["name"].as_str().unwrap()));
    }

    if is_enum_type(key_type) {
        return Some("uint256".to_string());
    }

    if is_contract_type(key_type) {
        return Some("address".to_string());
    }

    None
}

fn function_definition_kind_to_function_type(kind: Option<&str>) -> ContractFunctionType {
    match kind {
        Some("constructor") => ContractFunctionType::Constructor,
        Some("fallback") => ContractFunctionType::Fallback,
        Some("receive") => ContractFunctionType::Receive,
        Some("freeFunction") => ContractFunctionType::FreeFunction,
        _ => ContractFunctionType::Function,
    }
}

fn ast_visibility_to_visibility(visibility: &str) -> ContractFunctionVisibility {
    match visibility {
        "private" => ContractFunctionVisibility::Private,
        "internal" => ContractFunctionVisibility::Internal,
        "public" => ContractFunctionVisibility::Public,
        _ => ContractFunctionVisibility::External,
    }
}

fn is_contract_type(param: &serde_json::Value) -> bool {
    (param
        .pointer("/typeName/nodeType")
        .and_then(serde_json::Value::as_str)
        == Some("UserDefinedTypeName")
        || param.get("nodeType").and_then(serde_json::Value::as_str) == Some("UserDefinedTypeName"))
        && param
            .pointer("/typeDescriptions/typeString")
            .and_then(serde_json::Value::as_str)
            .map_or(false, |s| s.starts_with("contract "))
}

fn is_enum_type(param: &serde_json::Value) -> bool {
    (param
        .pointer("/typeName/nodeType")
        .and_then(serde_json::Value::as_str)
        == Some("UserDefinedTypeName")
        || param.get("nodeType").and_then(serde_json::Value::as_str) == Some("UserDefinedTypeName"))
        && param
            .pointer("/typeDescriptions/typeString")
            .and_then(serde_json::Value::as_str)
            .map_or(false, |s| s.starts_with("enum "))
}

fn is_elementary_type(param: &serde_json::Value) -> bool {
    param["nodeType"] == "ElementaryTypeName" || param["type"] == "ElementaryTypeName"
}

fn to_canonical_abi_type(type_: &str) -> String {
    if type_.starts_with("int[") {
        return format!("int256{}", &type_[3..]);
    }
    if type_ == "int" {
        return "int256".to_string();
    }
    if type_.starts_with("uint[") {
        return format!("uint256{}", &type_[4..]);
    }
    if type_ == "uint" {
        return "uint256".to_string();
    }
    if type_.starts_with("fixed[") {
        return format!("fixed128x128{}", &type_[5..]);
    }
    if type_ == "fixed" {
        return "fixed128x128".to_string();
    }
    if type_.starts_with("ufixed[") {
        return format!("ufixed128x128{}", &type_[6..]);
    }
    if type_ == "ufixed" {
        return "ufixed128x128".to_string();
    }

    type_.to_owned()
}

fn ast_src_to_source_location(
    src: &str,
    build_model_sources: &Arc<BuildModelSources>,
) -> anyhow::Result<Option<Arc<SourceLocation>>> {
    let parts: Vec<&str> = src.split(':').collect();
    if parts.len() != 3 {
        return Ok(None);
    }

    let offset = parts[0]
        .parse::<u32>()
        .with_context(|| format!("Failed to parse offset: {src:?}"))?;
    let length = parts[1]
        .parse::<u32>()
        .with_context(|| format!("Failed to parse length: {src:?}"))?;
    let file_id = parts[2]
        .parse::<u32>()
        .with_context(|| format!("Failed to parse file ID: {src:?}"))?;

    if build_model_sources.get(&file_id).is_none() {
        return Err(anyhow::anyhow!("Failed to find file by ID: {file_id}"));
    }

    Ok(Some(Arc::new(SourceLocation::new(
        Arc::clone(build_model_sources),
        file_id,
        offset,
        length,
    ))))
}

fn correct_selectors(
    bytecodes: &[ContractMetadata],
    compiler_output: &CompilerOutput,
) -> anyhow::Result<()> {
    for bytecode in bytecodes.iter().filter(|b| !b.is_deployment) {
        let mut contract = bytecode.contract.write();
        // Fetch the method identifiers for the contract from the compiler output
        let method_identifiers = match compiler_output
            .contracts
            .get(&contract.location.file().read().source_name)
            .and_then(|file| file.get(&contract.name))
            .map(|contract| &contract.evm.method_identifiers)
        {
            Some(ids) => ids,
            None => continue,
        };

        for (signature, hex_selector) in method_identifiers {
            let function_name = signature.split('(').next().unwrap_or("");
            let selector = hex::decode(hex_selector)
                .with_context(|| format!("Failed to decode hex: {hex_selector:?}"))?;

            let contract_function = contract.get_function_from_selector(&selector);

            if contract_function.is_some() {
                continue;
            }

            // NOTE: This code path is not covered by any of the existing tests.
            // Let's create a stack trace that exercises that code path or
            // let's remove it if/when we adapt our model to also properly
            // support ABI v2.
            let fixed_selector =
                contract.correct_selector(function_name.to_string(), selector.clone());

            if !fixed_selector {
                return Err(anyhow::anyhow!(
                "Failed to fix up the selector for one or more implementations of {}#{}. Hardhat Network can automatically fix this problem if you don't use function overloading.",
                contract.name, function_name
              ));
            }
        }
    }
    Ok(())
}

fn abi_method_id(name: &str, param_types: Vec<impl AsRef<str>>) -> Vec<u8> {
    let sig = format!(
        "{name}({})",
        // wasteful, but it's fine for now
        param_types
            .into_iter()
            .map(|x| to_canonical_abi_type(x.as_ref()))
            .collect::<Vec<_>>()
            .join(",")
    );
    let sig = sig.as_bytes();
    let sig = keccak256(sig);
    sig[..4].to_vec()
}

fn decode_evm_bytecode(
    contract: Arc<RwLock<Contract>>,
    solc_version: String,
    is_deployment: bool,
    compiler_bytecode: &CompilerOutputBytecode,
    build_model: &Arc<BuildModel>,
) -> anyhow::Result<ContractMetadata> {
    let library_address_positions = get_library_address_positions(compiler_bytecode);

    let immutable_references = compiler_bytecode
        .immutable_references
        .as_ref()
        .map(|refs| {
            refs.values()
                .flatten()
                .copied()
                .map(Into::into)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let normalized_code = normalize_compiler_output_bytecode(
        compiler_bytecode.object.clone(),
        &library_address_positions,
    )
    .with_context(|| format!("Failed to decode hex: {compiler_bytecode:?}"))?;

    let instructions = decode_instructions(
        &normalized_code,
        &compiler_bytecode.source_map,
        build_model,
        is_deployment,
    );

    Ok(ContractMetadata::new(
        Arc::clone(&build_model.file_id_to_source_file),
        contract,
        is_deployment,
        normalized_code,
        instructions,
        library_address_positions,
        immutable_references,
        solc_version,
    ))
}

fn decode_bytecodes(
    solc_version: String,
    compiler_output: &CompilerOutput,
    build_model: &Arc<BuildModel>,
) -> anyhow::Result<Vec<ContractMetadata>> {
    let mut bytecodes = Vec::new();

    for contract in build_model.contract_id_to_contract.values() {
        let contract_rc = contract.clone();

        let contract_evm_output = {
            let mut contract = contract.write();

            let contract_file = &contract.location.file().read().source_name.clone();
            let contract_evm_output = &compiler_output.contracts[contract_file][&contract.name].evm;
            let contract_abi_output = &compiler_output.contracts[contract_file][&contract.name].abi;

            for item in contract_abi_output {
                if item.r#type.as_deref() == Some("error") {
                    if let Ok(custom_error) = CustomError::from_abi(item.clone()) {
                        contract.add_custom_error(custom_error);
                    }
                }
            }

            contract_evm_output
        };

        // This is an abstract contract
        if contract_evm_output.bytecode.object.is_empty() {
            continue;
        }

        let deployment_bytecode = decode_evm_bytecode(
            contract_rc.clone(),
            solc_version.clone(),
            true,
            &contract_evm_output.bytecode,
            build_model,
        )?;

        let runtime_bytecode = decode_evm_bytecode(
            contract_rc.clone(),
            solc_version.clone(),
            false,
            &contract_evm_output.deployed_bytecode,
            build_model,
        )?;

        bytecodes.push(deployment_bytecode);
        bytecodes.push(runtime_bytecode);
    }

    Ok(bytecodes)
}
