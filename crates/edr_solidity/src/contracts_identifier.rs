//! A data structure that allows searching for well-known bytecodes and their
//! contracts.
//!
//! In addition to being a trie, it also performs normalization of the bytecode
//! for the libraries by zeroing out the addresses of link references, i.e. the
//! addresses of the libraries or immutable references for the lookup.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use edr_eth::{bytecode::opcode::OpCode, Address};

use crate::build_model::ContractMetadata;

/// The result of searching for a bytecode in a [`BytecodeTrie`].
enum TrieSearch<'a> {
    /// An exact match was found.
    ExactHit(Arc<ContractMetadata>),
    /// No exact match found; a node with the longest prefix is returned.
    LongestPrefixNode(&'a BytecodeTrie),
}

/// This class represent a somewhat special Trie of bytecodes.
///
/// What makes it special is that every node has a set of all of its descendants
/// and its depth.
#[derive(Debug, Clone)]
struct BytecodeTrie {
    child_nodes: HashMap<u8, Box<BytecodeTrie>>,
    descendants: Vec<Arc<ContractMetadata>>,
    match_: Option<Arc<ContractMetadata>>,
    depth: Option<u32>,
}

impl BytecodeTrie {
    fn new(depth: Option<u32>) -> BytecodeTrie {
        BytecodeTrie {
            child_nodes: HashMap::new(),
            descendants: Vec::new(),
            match_: None,
            depth,
        }
    }

    fn add(&mut self, bytecode: Arc<ContractMetadata>) {
        let mut cursor = self;

        let bytecode_normalized_code = &bytecode.normalized_code;
        for (index, byte) in bytecode_normalized_code.iter().enumerate() {
            cursor.descendants.push(bytecode.clone());

            let node = cursor
                .child_nodes
                .entry(*byte)
                .or_insert_with(|| Box::new(BytecodeTrie::new(Some(index as u32))));

            cursor = node;
        }

        // If multiple contracts with the exact same bytecode are added we keep the last
        // of them. Note that this includes the metadata hash, so the chances of
        // happening are pretty remote, except in super artificial cases that we
        // have in our test suite.
        cursor.match_ = Some(bytecode.clone());
    }

    /// Searches for a bytecode. If it's an exact match, it is returned. If
    /// there's no match, but a prefix of the code is found in the trie, the
    /// node of the longest prefix is returned. If the entire code is
    /// covered by the trie, and there's no match, we return None.
    fn search(&self, code: &[u8], current_code_byte: u32) -> Option<TrieSearch<'_>> {
        if current_code_byte > code.len() as u32 {
            return None;
        }

        let mut cursor = self;

        for byte in code.iter().skip(current_code_byte as usize) {
            let child_node = cursor.child_nodes.get(byte);

            if let Some(node) = child_node {
                cursor = node;
            } else {
                return Some(TrieSearch::LongestPrefixNode(cursor));
            }
        }

        cursor
            .match_
            .as_ref()
            .map(|bytecode| TrieSearch::ExactHit(bytecode.clone()))
    }
}

/// Returns true if the `last_byte` is placed right when the metadata starts or
/// after it.
fn is_matching_metadata(code: &[u8], last_byte: u32) -> bool {
    let mut byte = 0;
    while byte < last_byte {
        // It's possible we don't recognize the opcode if it's from an unknown chain, so
        // just return false in that case.
        let Some(opcode) = OpCode::new(code[byte as usize]) else {
            return false;
        };

        let next = code.get(byte as usize + 1).copied().and_then(OpCode::new);

        if opcode == OpCode::REVERT && next == Some(OpCode::INVALID) {
            return true;
        }

        byte += 1 + u32::from(opcode.info().immediate_size());
    }

    false
}

/// A data structure that allows searching for well-known bytecodes.
#[derive(Debug)]
pub struct ContractsIdentifier {
    trie: BytecodeTrie,
    cache: HashMap<Vec<u8>, Arc<ContractMetadata>>,
    enable_cache: bool,
}

impl Default for ContractsIdentifier {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ContractsIdentifier {
    /// Creates a new [`ContractsIdentifier`].
    pub fn new(enable_cache: Option<bool>) -> ContractsIdentifier {
        let enable_cache = enable_cache.unwrap_or(true);

        ContractsIdentifier {
            trie: BytecodeTrie::new(None),
            cache: HashMap::new(),
            enable_cache,
        }
    }

    /// Adds a bytecode to the tree.
    pub fn add_bytecode(&mut self, bytecode: Arc<ContractMetadata>) {
        self.trie.add(bytecode);
        self.cache.clear();
    }

    fn search_bytecode_from_root(
        &mut self,
        is_create: bool,
        code: &[u8],
    ) -> Option<Arc<ContractMetadata>> {
        let normalize_libraries = true;
        let first_byte_to_search = 0;

        Self::search_bytecode_at_depth(
            is_create,
            code,
            normalize_libraries,
            &self.trie,
            first_byte_to_search,
        )
    }

    fn search_bytecode_at_depth(
        is_create: bool,
        code: &[u8],
        normalize_libraries: bool,
        trie: &BytecodeTrie,
        first_byte_to_search: u32,
    ) -> Option<Arc<ContractMetadata>> {
        let search_result = match trie.search(code, first_byte_to_search) {
            None => return None,
            Some(TrieSearch::ExactHit(bytecode)) => return Some(bytecode.clone()),
            Some(TrieSearch::LongestPrefixNode(trie)) => trie,
        };

        // Deployment messages have their abi-encoded arguments at the end of the
        // bytecode.
        //
        // We don't know how long those arguments are, as we don't know which contract
        // is being deployed, hence we don't know the signature of its
        // constructor.
        //
        // To make things even harder, we can't trust that the user actually passed the
        // right amount of arguments.
        //
        // Luckily, the chances of a complete deployment bytecode being the prefix of
        // another one are remote. For example, most of the time it ends with
        // its metadata hash, which will differ.
        //
        // We take advantage of this last observation, and just return the bytecode that
        // exactly matched the search_result (sub)trie that we got.
        match &search_result.match_ {
            Some(bytecode) if is_create && bytecode.is_deployment => {
                return Some(bytecode.clone());
            }
            _ => {}
        };

        if normalize_libraries {
            for bytecode_with_libraries in &search_result.descendants {
                if bytecode_with_libraries.library_address_positions.is_empty()
                    && bytecode_with_libraries.immutable_references.is_empty()
                {
                    continue;
                }

                let mut normalized_code = code.to_vec();
                // zero out addresses
                for &pos in &bytecode_with_libraries.library_address_positions {
                    let range = pos as usize..(pos as usize + Address::len_bytes());

                    if let Some(chunk) = normalized_code.get_mut(range) {
                        chunk.fill(0);
                    }
                }
                // zero out slices
                for imm in &bytecode_with_libraries.immutable_references {
                    let range = imm.start as usize..(imm.start as usize + imm.length as usize);

                    if let Some(chunk) = normalized_code.get_mut(range) {
                        chunk.fill(0);
                    }
                }

                let normalized_result = Self::search_bytecode_at_depth(
                    is_create,
                    &normalized_code,
                    false,
                    search_result,
                    search_result.depth.map_or(0, |depth| depth + 1),
                );

                if normalized_result.is_some() {
                    return normalized_result;
                }
            }
        }

        // If we got here we may still have the contract, but with a different metadata
        // hash.
        //
        // We check if we got to match the entire executable bytecode, and are just
        // stuck because of the metadata. If that's the case, we can assume that
        // any descendant will be a valid Bytecode, so we just choose the most
        // recently added one.
        //
        // The reason this works is because there's no chance that Solidity includes an
        // entire bytecode (i.e. with metadata), as a prefix of another one.
        if let Some(search_depth) = search_result.depth {
            if is_matching_metadata(code, search_depth) && !search_result.descendants.is_empty() {
                return Some(
                    search_result.descendants[search_result.descendants.len() - 1].clone(),
                );
            }
        }

        None
    }

    /// Searches for a bytecode that matches the given (call/create) code.
    pub fn get_bytecode_for_call(
        &mut self,
        code: &[u8],
        is_create: bool,
    ) -> Option<Arc<ContractMetadata>> {
        let normalized_code = normalize_library_runtime_bytecode_if_necessary(code);

        if self.enable_cache {
            let cached = self.cache.get(&*normalized_code);

            if let Some(cached) = cached {
                return Some(cached.clone());
            }
        }

        let result = self.search_bytecode_from_root(is_create, &normalized_code);

        if self.enable_cache {
            if let Some(result) = &result {
                if !self.cache.contains_key(&*normalized_code) {
                    self.cache.insert(normalized_code.to_vec(), result.clone());
                }
            }
        }

        result
    }
}

fn normalize_library_runtime_bytecode_if_necessary(bytecode: &[u8]) -> Cow<'_, [u8]> {
    let mut bytecode = Cow::Borrowed(bytecode);

    // Libraries' protection normalization:
    // Solidity 0.4.20 introduced a protection to prevent libraries from being
    // called directly. This is done by modifying the code on deployment, and
    // hard-coding the contract address. The first instruction is a PUSH20 of
    // the address, which we zero-out as a way of normalizing it. Note that it's
    // also zeroed-out in the compiler output.
    if bytecode.first().copied() == Some(OpCode::PUSH20.get()) {
        bytecode.to_mut()[1..][..Address::len_bytes()].fill(0);
    }

    bytecode
}

#[cfg(test)]
mod tests {
    use std::vec;

    use parking_lot::RwLock;

    use super::*;
    use crate::{
        artifacts::ImmutableReference,
        build_model::{Contract, ContractKind, SourceFile, SourceLocation},
    };

    fn create_sources() -> Arc<HashMap<u32, Arc<RwLock<SourceFile>>>> {
        let mut sources = HashMap::new();
        let file = Arc::new(RwLock::new(SourceFile::new(
            "test.sol".to_string(),
            "".to_string(),
        )));

        sources.insert(0, file.clone());

        Arc::new(sources)
    }

    fn create_test_contract() -> Arc<RwLock<Contract>> {
        let sources = create_sources();

        let location = Arc::new(SourceLocation::new(sources.clone(), 0, 0, 0));

        Arc::new(RwLock::new(Contract::new(
            "TestContract".to_string(),
            ContractKind::Contract,
            location,
        )))
    }

    fn create_test_bytecode(normalized_code: Vec<u8>) -> Arc<ContractMetadata> {
        let sources = create_sources();
        let contract = create_test_contract();
        let is_deployment = false;

        let instructions = vec![];
        let library_offsets = vec![];
        let immutable_references = vec![];

        Arc::new(ContractMetadata::new(
            sources,
            contract,
            is_deployment,
            normalized_code,
            instructions,
            library_offsets,
            immutable_references,
            "<dummy-version>".to_string(),
        ))
    }

    fn create_test_deployment_bytecode(normalized_code: Vec<u8>) -> Arc<ContractMetadata> {
        let sources = create_sources();
        let contract = create_test_contract();
        let is_deployment = true;

        let instructions = vec![];
        let library_offsets = vec![];
        let immutable_references = vec![];

        Arc::new(ContractMetadata::new(
            sources,
            contract,
            is_deployment,
            normalized_code,
            instructions,
            library_offsets,
            immutable_references,
            "<dummy-version>".to_string(),
        ))
    }

    fn create_test_bytecode_with_libraries_and_immutable_references(
        normalized_code: Vec<u8>,
        library_offsets: Vec<u32>,
        immutable_references: Vec<ImmutableReference>,
    ) -> Arc<ContractMetadata> {
        let sources = create_sources();
        let contract = create_test_contract();
        let is_deployment = false;

        let instructions = vec![];

        Arc::new(ContractMetadata::new(
            sources,
            contract,
            is_deployment,
            normalized_code,
            instructions,
            library_offsets,
            immutable_references,
            "<dummy-version>".to_string(),
        ))
    }

    #[test]
    fn test_contracts_identifier_empty() {
        let mut contracts_identifier = ContractsIdentifier::default();

        // should not find any bytecode for a call trace
        let is_create = false;
        let contract = contracts_identifier.search_bytecode_from_root(is_create, &[1, 2, 3, 4, 5]);
        assert!(contract.is_none());

        // should not find any bytecode for a create trace
        let is_create = true;
        let contract = contracts_identifier.search_bytecode_from_root(is_create, &[1, 2, 3, 4, 5]);
        assert!(contract.is_none());
    }

    #[test]
    fn test_contracts_identifier_single_matching_bytecode() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode(vec![1, 2, 3, 4, 5]);
        contracts_identifier.add_bytecode(bytecode.clone());

        // should find a bytecode that matches exactly
        let is_create = false;
        let contract = contracts_identifier.search_bytecode_from_root(is_create, &[1, 2, 3, 4, 5]);
        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );

        // should not find a bytecode that doesn't match
        let is_create = false;
        let contract = contracts_identifier.search_bytecode_from_root(is_create, &[1, 2, 3, 4, 6]);
        assert!(contract.is_none());
    }

    #[test]
    fn test_contracts_identifier_multiple_matches_same_prefix() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode1 = create_test_bytecode(vec![1, 2, 3, 4, 5]);
        let bytecode2 = create_test_bytecode(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        contracts_identifier.add_bytecode(bytecode1.clone());
        contracts_identifier.add_bytecode(bytecode2.clone());

        // should find the exact match
        let contract = contracts_identifier.search_bytecode_from_root(false, &[1, 2, 3, 4, 5]);
        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode1))
        );

        // should find the exact match
        let contract =
            contracts_identifier.search_bytecode_from_root(false, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode2))
        );

        // should not find a bytecode that doesn't match
        let contract =
            contracts_identifier.search_bytecode_from_root(false, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(contract.is_none());
    }

    #[test]
    fn test_contracts_identifier_trace_matches_common_prefix() {
        let mut contracts_identifier = ContractsIdentifier::default();

        // add two bytecodes that share a prefix
        let bytecode1 = create_test_bytecode(vec![1, 2, 3, 4, 5]);
        let bytecode2 = create_test_bytecode(vec![1, 2, 3, 6, 7]);
        contracts_identifier.add_bytecode(bytecode1.clone());
        contracts_identifier.add_bytecode(bytecode2.clone());

        // search a trace that matches the common prefix
        let contract = contracts_identifier.search_bytecode_from_root(false, &[1, 2, 3]);
        assert!(contract.is_none());
    }

    #[test]
    fn test_contracts_identifier_trace_matches_deployment_bytecode_prefix() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_deployment_bytecode(vec![1, 2, 3, 4, 5]);
        contracts_identifier.add_bytecode(bytecode.clone());

        // a create trace that matches the a deployment bytecode plus some extra stuff
        // (constructor args)
        let is_create = true;
        let contract =
            contracts_identifier.search_bytecode_from_root(is_create, &[1, 2, 3, 4, 5, 10, 11]);
        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );

        // the same bytecode, but for a call trace, should not match
        let contract =
            contracts_identifier.search_bytecode_from_root(false, &[1, 2, 3, 4, 5, 10, 11]);
        assert!(contract.is_none());

        // the same scenario but with a runtime bytecode shouldn't result in matches
        let mut contracts_identifier = ContractsIdentifier::default();
        let bytecode = create_test_bytecode(vec![1, 2, 3, 4, 5]);
        contracts_identifier.add_bytecode(bytecode.clone());

        let contract =
            contracts_identifier.search_bytecode_from_root(true, &[1, 2, 3, 4, 5, 10, 11]);
        assert!(contract.is_none());

        let contract =
            contracts_identifier.search_bytecode_from_root(false, &[1, 2, 3, 4, 5, 10, 11]);
        assert!(contract.is_none());
    }

    #[test]
    fn test_contracts_identifier_bytecode_with_one_library() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode_with_libraries_and_immutable_references(
            vec![
                // 0 -------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, library address
                // -------------------------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 40 ------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
            ],
            vec![20],
            vec![],
        );
        contracts_identifier.add_bytecode(bytecode.clone());

        // the same bytecode, but for a call trace, should not match
        let contract = contracts_identifier.search_bytecode_from_root(
            false,
            &[
                // 0 -----------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, library address
                // -----------------------------------------------------------------
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                118, 119, 120,
                // 40 ----------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
            ],
        );

        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }

    #[test]
    fn test_contracts_identifier_bytecode_with_one_immutable_reference() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode_with_libraries_and_immutable_references(
            vec![
                // 0 -------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // --------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 30 ------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
            ],
            vec![],
            vec![ImmutableReference {
                start: 20,
                length: 10,
            }],
        );
        contracts_identifier.add_bytecode(bytecode.clone());

        // the same bytecode, but for a call trace, should not match
        let contract = contracts_identifier.search_bytecode_from_root(
            false,
            &[
                // 0 -----------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // ------------------------------------------------
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                // 30 ----------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
            ],
        );

        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }

    #[test]
    fn test_contracts_identifier_bytecode_with_one_library_and_one_immutable_reference() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode_with_libraries_and_immutable_references(
            vec![
                // 0 -------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // --------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 30 ------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
                // 35, library address
                // -------------------------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 55 ------------------------------------------------------------------------------
                26, 27, 28, 29, 30,
            ],
            vec![35],
            vec![ImmutableReference {
                start: 20,
                length: 10,
            }],
        );
        contracts_identifier.add_bytecode(bytecode.clone());

        // the same bytecode, but for a call trace, should not match
        let contract = contracts_identifier.search_bytecode_from_root(
            false,
            &[
                // 0 -----------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // ------------------------------------------------
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                // 30 ----------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
                // 35, library address
                // -----------------------------------------------------------------
                201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                218, 219, 220,
                // 55 ----------------------------------------------------------------------------------
                26, 27, 28, 29, 30,
            ],
        );

        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }

    #[test]
    fn test_contracts_identifier_bytecode_with_multiple_libraries_and_immutable_references() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode_with_libraries_and_immutable_references(
            vec![
                // 0 -------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // --------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 30 ------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
                // 35, library address
                // -------------------------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 55 ------------------------------------------------------------------------------
                26, 27, 28, 29, 30,
                // 60, another library address
                // -----------------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // 80 ------------------------------------------------------------------------------
                31, 32, 33, 34, 35,
                // 85, immutable reference of length 30
                // --------------------------------------------
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
                // 115 -----------------------------------------------------------------------------
                36, 37, 38, 39, 40,
            ],
            vec![35, 60],
            vec![
                ImmutableReference {
                    start: 20,
                    length: 10,
                },
                ImmutableReference {
                    start: 85,
                    length: 30,
                },
            ],
        );
        contracts_identifier.add_bytecode(bytecode.clone());

        // the same bytecode, but for a call trace, should not match
        let contract = contracts_identifier.search_bytecode_from_root(
            false,
            &[
                // 0 -----------------------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                // 20, immutable reference of length 10
                // ------------------------------------------------
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                // 30 ----------------------------------------------------------------------------------
                21, 22, 23, 24, 25,
                // 35, library address
                // -----------------------------------------------------------------
                201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                218, 219, 220,
                // 55 ----------------------------------------------------------------------------------
                26, 27, 28, 29, 30,
                // 60, another library address
                // ---------------------------------------------------------
                221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
                238, 239, 240,
                // 80 ----------------------------------------------------------------------------------
                31, 32, 33, 34, 35,
                // 85, immutable reference of length 30
                // ------------------------------------------------
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                // 115 ---------------------------------------------------------------------------------
                36, 37, 38, 39, 40,
            ],
        );

        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }

    #[test]
    fn test_contracts_identifier_bytecode_with_different_metadata() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode(vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            // metadata ----------------------------------------------------------------------------
            0xfd, 0xfe, 11, 12, 13, 14, 15,
        ]);
        contracts_identifier.add_bytecode(bytecode.clone());

        let contract = contracts_identifier.search_bytecode_from_root(
            false,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                // metadata ----------------------------------------------------------------------------
                0xfd, 0xfe, 21, 22, 23,
            ],
        );
        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }

    #[test]
    fn test_contracts_identifier_normalized_library_runtime_code() {
        let mut contracts_identifier = ContractsIdentifier::default();

        let bytecode = create_test_bytecode(vec![
            // PUSH20
            0x73,
            // library address
            // ---------------------------------------------------------------------
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // rest of the code
            // --------------------------------------------------------------------
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        ]);
        contracts_identifier.add_bytecode(bytecode.clone());

        let contract = contracts_identifier.get_bytecode_for_call(
            &[
                // PUSH20
                0x73,
                // library address
                // ---------------------------------------------------------------------
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                // rest of the code
                // --------------------------------------------------------------------
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            ],
            false,
        );

        assert_eq!(
            contract.as_ref().map(Arc::as_ptr),
            Some(Arc::as_ptr(&bytecode))
        );
    }
}
