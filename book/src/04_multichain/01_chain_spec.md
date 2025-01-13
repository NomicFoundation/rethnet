# Chain Specification

EDR uses a concept called _chain specification_ to define all necessary types and functionality to support an Ethereum chain at compile-time.

This is achieved through the usage of multiple traits, some of which are supertraits of each other, providing increasing scope of functionality in the EDR ecosystem:

- Revm primitives: `revm_primitives::EvmWiring`
  - Implements the main trait for implementing associated chain-specific primitive types.
- Revm: `revm::EvmWiring`
  - Adds an additional function to `revm_primitives::EvmWiring` that depends on the `Evm`.
- Header builder: `edr_eth::EthHeaderConstants`
- RPC client: `edr_rpc_eth::RpcSpec`
- EVM runtime: `edr_evm::RuntimeSpec`
- EVM provider: `edr_provider::ProviderSpec`
- EDR N-API bindings: `edr_napi::SyncNapiSpec`

Most of these traits have a `Sync*` equivalent (e.g. `SyncRuntimeSpec`) which is automatically implemented for types that are `Send` and `Sync`.

## Supported Chain Types

Currently, EDR supports the following chain types out-of-the-box.

- L1 Ethereum
- Optimism

## Adding your own Chain Type

To add a new chain type, add a new unit struct. E.g.

```rs
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, RlpEncodable)]
pub struct GenericChainSpec;
```

Depending on which functionality you want to support for your chain type, you can implement some or all of the traits outlined [above](#chain-specification).
