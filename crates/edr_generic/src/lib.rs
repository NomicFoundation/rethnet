//! A slightly more flexible chain specification for Ethereum Layer 1 chain.

mod eip2718;
mod receipt;
mod rpc;
mod spec;
mod transaction;

/// The chain specification for Ethereum Layer 1 that is a bit more lenient
/// and allows for more flexibility in contrast to
/// [`L1ChainSpec`](edr_eth::l1::L1ChainSpec).
///
/// Specifically:
/// - it allows unknown transaction types (treats them as legacy
///   [`Eip155`](edr_eth::transaction::signed::Eip155) transactions)
/// - it allows remote blocks with missing `nonce` and `mix_hash` fields
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, alloy_rlp::RlpEncodable)]
pub struct GenericChainSpec;
