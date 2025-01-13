#![cfg(feature = "test-remote")]

use edr_evm::impl_full_block_tests;
use edr_optimism::OptimismChainSpec;
use edr_test_utils::env::get_alchemy_url;

impl_full_block_tests! {
    mainnet_regolith => OptimismChainSpec {
        block_number: 105_235_064,
        url: get_alchemy_url().replace("eth-", "opt-"),
    },
    mainnet_canyon => OptimismChainSpec {
        block_number: 115_235_064,
        url: get_alchemy_url().replace("eth-", "opt-"),
    },
    mainnet_ecotone => OptimismChainSpec {
        block_number: 121_874_088,
        url: get_alchemy_url().replace("eth-", "opt-"),
    },
    mainnet_fjord => OptimismChainSpec {
        block_number: 122_514_212,
        url: get_alchemy_url().replace("eth-", "opt-"),
    },
}
