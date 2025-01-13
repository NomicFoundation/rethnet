use std::sync::OnceLock;

use edr_eth::{l1, HashMap};

use super::{Activations, ChainConfig, ForkCondition};

const MAINNET_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(0), l1::SpecId::FRONTIER),
    (ForkCondition::Block(200_000), l1::SpecId::FRONTIER_THAWING),
    (ForkCondition::Block(1_150_000), l1::SpecId::HOMESTEAD),
    (ForkCondition::Block(1_920_000), l1::SpecId::DAO_FORK),
    (ForkCondition::Block(2_463_000), l1::SpecId::TANGERINE),
    (ForkCondition::Block(2_675_000), l1::SpecId::SPURIOUS_DRAGON),
    (ForkCondition::Block(4_370_000), l1::SpecId::BYZANTIUM),
    (ForkCondition::Block(7_280_000), l1::SpecId::CONSTANTINOPLE),
    (ForkCondition::Block(7_280_000), l1::SpecId::PETERSBURG),
    (ForkCondition::Block(9_069_000), l1::SpecId::ISTANBUL),
    (ForkCondition::Block(9_200_000), l1::SpecId::MUIR_GLACIER),
    (ForkCondition::Block(12_244_000), l1::SpecId::BERLIN),
    (ForkCondition::Block(12_965_000), l1::SpecId::LONDON),
    (ForkCondition::Block(13_773_000), l1::SpecId::ARROW_GLACIER),
    (ForkCondition::Block(15_050_000), l1::SpecId::GRAY_GLACIER),
    (ForkCondition::Block(15_537_394), l1::SpecId::MERGE),
    (ForkCondition::Block(17_034_870), l1::SpecId::SHANGHAI),
    (ForkCondition::Block(19_426_589), l1::SpecId::CANCUN),
];

fn mainnet_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = MAINNET_HARDFORKS.into();

        ChainConfig {
            name: "mainnet".to_string(),
            hardfork_activations,
        }
    })
}

const ROPSTEN_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(1_700_000), l1::SpecId::BYZANTIUM),
    (ForkCondition::Block(4_230_000), l1::SpecId::CONSTANTINOPLE),
    (ForkCondition::Block(4_939_394), l1::SpecId::PETERSBURG),
    (ForkCondition::Block(6_485_846), l1::SpecId::ISTANBUL),
    (ForkCondition::Block(7_117_117), l1::SpecId::MUIR_GLACIER),
    (ForkCondition::Block(9_812_189), l1::SpecId::BERLIN),
    (ForkCondition::Block(10_499_401), l1::SpecId::LONDON),
];

fn ropsten_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = ROPSTEN_HARDFORKS.into();

        ChainConfig {
            name: "ropsten".to_string(),
            hardfork_activations,
        }
    })
}

const RINKEBY_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(1_035_301), l1::SpecId::BYZANTIUM),
    (ForkCondition::Block(3_660_663), l1::SpecId::CONSTANTINOPLE),
    (ForkCondition::Block(4_321_234), l1::SpecId::PETERSBURG),
    (ForkCondition::Block(5_435_345), l1::SpecId::ISTANBUL),
    (ForkCondition::Block(8_290_928), l1::SpecId::BERLIN),
    (ForkCondition::Block(8_897_988), l1::SpecId::LONDON),
];

fn rinkeby_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = RINKEBY_HARDFORKS.into();

        ChainConfig {
            name: "rinkeby".to_string(),
            hardfork_activations,
        }
    })
}

const GOERLI_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(0), l1::SpecId::PETERSBURG),
    (ForkCondition::Block(1_561_651), l1::SpecId::ISTANBUL),
    (ForkCondition::Block(4_460_644), l1::SpecId::BERLIN),
    (ForkCondition::Block(5_062_605), l1::SpecId::LONDON),
    (ForkCondition::Block(7_382_818), l1::SpecId::MERGE),
    (ForkCondition::Block(8_656_123), l1::SpecId::SHANGHAI),
    (ForkCondition::Block(10_388_176), l1::SpecId::CANCUN),
];

fn goerli_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = GOERLI_HARDFORKS.into();

        ChainConfig {
            name: "goerli".to_string(),
            hardfork_activations,
        }
    })
}

const KOVAN_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(5_067_000), l1::SpecId::BYZANTIUM),
    (ForkCondition::Block(9_200_000), l1::SpecId::CONSTANTINOPLE),
    (ForkCondition::Block(10_255_201), l1::SpecId::PETERSBURG),
    (ForkCondition::Block(14_111_141), l1::SpecId::ISTANBUL),
    (ForkCondition::Block(24_770_900), l1::SpecId::BERLIN),
    (ForkCondition::Block(26_741_100), l1::SpecId::LONDON),
];

fn kovan_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = KOVAN_HARDFORKS.into();

        ChainConfig {
            name: "kovan".to_string(),
            hardfork_activations,
        }
    })
}

const HOLESKY_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(0), l1::SpecId::MERGE),
    (ForkCondition::Block(6_698), l1::SpecId::SHANGHAI),
    (ForkCondition::Block(894_733), l1::SpecId::CANCUN),
];

fn holesky_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = HOLESKY_HARDFORKS.into();

        ChainConfig {
            name: "holesky".to_string(),
            hardfork_activations,
        }
    })
}

const SEPOLIA_HARDFORKS: &[(ForkCondition, l1::SpecId)] = &[
    (ForkCondition::Block(0), l1::SpecId::LONDON),
    (ForkCondition::Block(1_450_409), l1::SpecId::MERGE),
    (ForkCondition::Block(2_990_908), l1::SpecId::SHANGHAI),
    (ForkCondition::Block(5_187_023), l1::SpecId::CANCUN),
];

fn sepolia_config() -> &'static ChainConfig<l1::SpecId> {
    static CONFIG: OnceLock<ChainConfig<l1::SpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = SEPOLIA_HARDFORKS.into();

        ChainConfig {
            name: "sepolia".to_string(),
            hardfork_activations,
        }
    })
}

fn chain_configs() -> &'static HashMap<u64, &'static ChainConfig<l1::SpecId>> {
    static CONFIGS: OnceLock<HashMap<u64, &'static ChainConfig<l1::SpecId>>> = OnceLock::new();

    CONFIGS.get_or_init(|| {
        let mut hardforks = HashMap::new();
        hardforks.insert(1, mainnet_config());
        hardforks.insert(3, ropsten_config());
        hardforks.insert(4, rinkeby_config());
        hardforks.insert(5, goerli_config());
        hardforks.insert(42, kovan_config());
        hardforks.insert(17_000, holesky_config());
        hardforks.insert(11_155_111, sepolia_config());

        hardforks
    })
}

/// Returns the name corresponding to the provided chain ID, if it is supported.
pub fn chain_name(chain_id: u64) -> Option<&'static str> {
    chain_configs()
        .get(&chain_id)
        .map(|config| config.name.as_str())
}

/// Returns the hardfork activations corresponding to the provided chain ID, if
/// it is supported.
pub fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<l1::SpecId>> {
    chain_configs()
        .get(&chain_id)
        .map(|config| &config.hardfork_activations)
}
