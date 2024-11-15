# @nomicfoundation/edr

## 0.6.5

### Patch Changes

- 9c062ad: fix: don't panic when a precompile errors

## 0.6.4

### Patch Changes

- 14620c9: Fix panic due to remote nodes not returning total difficulty by adding fallback value

## 0.6.3

### Patch Changes

- a1cee4f: fix(tracing): Make sure to correctly 0x-prefix hex encoded strings in custom error message
- 7047e6c: Fixed panic from unknown opcode in stack traces

## 0.6.2

### Patch Changes

- af56289: fix(tracing): Decode unmapped instructions only at the opcode boundary
- debac88: fix(tracing): Use correct subtrace when detecting error error propagation across delegatecall

## 0.6.1

### Patch Changes

- c8fbf3b: Allow forked block time offset to be negative
- bb3808b: Handle optional solc settings better in the tracing engine

## 0.6.0

### Minor Changes

- a417e19: Renamed `json` field in response to `data` and changed its type to `string | object` (specified as `string | any` due to an napi-rs limitation).

### Patch Changes

- 2f1c189: Improved error message and added option to skip unsupported transaction type in `debug_traceTransaction`. Set `__EDR_UNSAFE_SKIP_UNSUPPORTED_TRANSACTION_TYPES=true` as an environment variable to enable this.

## 0.5.2

### Patch Changes

- 66ca796: fix: added `json` alias property in provider response to fix breaking change in v0.5.1

## 0.5.1

### Patch Changes

- 0bba027: Fixed a bug in fork mode where the locally overridden chain id was used instead of the remote chain id when simulating transactions in pre-fork blocks.
- e02eb22: Fixed a bug in the JSON-RPC where we previously allowed a null value for storage keys of an access list
- 8ae31b9: Fixed crash when returning large JSON responses

## 0.5.0

### Minor Changes

- 07c7667: Added support for RIP-7212 by enabling it in the provider configuration

## 0.4.2

### Patch Changes

- b056511: Fixes restriction to disallow remote blocks without a nonce
- e650c9e: Upgraded revm to v37

## 0.4.1

### Patch Changes

- 245fd07: Provide a more helpful message when passing a timestamp bigger than 2^64 via JSON-RPC
- d87a8c4: Fixed a problem that prevented forking chains without mix hash fields in their blocks
- 5984f11: Fixed missing remote contract code when setting storage

## 0.4.0

### Minor Changes

- 5b4acdd: Added verbose tracing for hardhat-tracer. Breaking change: The `stack_top` property of `edr_napi::trace::TracingStep` was removed and `stack` was added instead. Please see the documentation of the struct for details.

### Patch Changes

- af8053f: Fixed order of parameters for eth_sign (#455)
- 32b765d: Added basic support for `eth_maxPriorityFeePerGas`, with a hardcoded response of 1 gwei (#369)

## 0.3.8

### Patch Changes

- a4d5b62: Added support for blob transactions (EIP-4844) when auto-mining is enabled and the mempool is empty
- ad1a4e8: Added support for EIP-712 payloads with signed integers
- b03c944: Upgraded revm to commit aceb093
- 3dfd1a5: Fixed `smock.fake` causing a panic
- 0db9db9: Added retry for sporadic "missing trie node" JSON-RPC errors
- c5da529: Fixed eth_call and eth_sendTransaction to allow passing input data with both `input` and `data` fields
- caa6f85: Added a workaround for an npm bug by using all builds as normal dependencies

## 0.3.7

_This version didn't include any changes, but it fixed a release issue that caused one of our supported architectures to not be published._

## 0.3.6

### Patch Changes

- e82246c: Removed usage of batch calls to prevent 429 errors
- 7745f35: Fixed mining of blocks when forking a chain that has non-Ethereum block headers
- 8928b77: Improved performance by avoiding call to ecrecover for known sender
- 1a83206: Improved performance - especially for large test suites - by changing internal data structures to achieve constant time checkpointing
- 7745f35: Fixed panic when re-setting global default subscriber
- 9be7023: Improved performance by avoiding clones of transaction traces

## 0.3.5

### Patch Changes

- e5f048e: Removed i686 & ARM builds of EDR
- 3d7f13e: Fixed a bug in `hardhat_metadata` where the local chain id was being used in the fork metadata
- bbc3a6d: Fixed missing KZG point evaluation precompile for Cancun hardfork
- 219f457: Fixed incorrect derivation of hardfork when executing a call in fork mode for the block immediately preceding the fork

## 0.3.4

### Patch Changes

- 71287a8: Removed API keys from RPC error messages
- bdf3971: Fixed 429 HTTP error by using smaller batches when querying genesis account info
- 7c23825: Fixed 429 HTTP error by increasing rate limiting retries
- 92693fb: Fixed calculation of used blob gas for post-Cancun blocks
- 62e28ad: Fixed eth_getLogs RPC request for pre-Merge hardforks

## 0.3.3

### Patch Changes

- 60b2a62: Fixed missing support for hex string as salt in `eth_signTypedData_v4`
- 3ac838b: Fixed detection of Cancun blocks on mainnet
- 7d0f981: Fixed node.js runtime freezing on shutdown

## 0.3.2

### Patch Changes

- b13f58a: Fixed failure when retrieving remote code during state modifications (#5000)
- 19eeeb9: Simplified internal set_account_storage_slot API (#5001)

## 0.3.1

### Patch Changes

- 591b7c5: Fixed failing RPC requests for certain providers due to missing content-type header (#4992)

## 0.3.0

### Minor Changes

- ac155d6: Bump Rust to v1.76

### Patch Changes

- 87da82b: Fixed a problem when forking networks with non-standard transaction types (#4963)
- 5fe1728: Fixed a bug in `hardhat_setStorageAt` that occured when the storage of a remote contract was modified during forking (#4970)
- 0ec305f: Fixed a bug in `hardhat_dropTransaction` where empty queues persisted and caused panics
- a5071e5: Fixed a bug in `eth_estimateGas` where a call would fail because the nonce was being checked
