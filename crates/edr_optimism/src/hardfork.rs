use std::sync::OnceLock;

use edr_eth::HashMap;
use edr_evm::hardfork::{Activations, ChainConfig, ForkCondition};

use crate::OptimismSpecId;

const MAINNET_HARDFORKS: &[(ForkCondition, OptimismSpecId)] = &[
    (ForkCondition::Block(0), OptimismSpecId::FRONTIER),
    (ForkCondition::Block(0), OptimismSpecId::HOMESTEAD),
    (ForkCondition::Block(0), OptimismSpecId::TANGERINE),
    (ForkCondition::Block(0), OptimismSpecId::SPURIOUS_DRAGON),
    (ForkCondition::Block(0), OptimismSpecId::BYZANTIUM),
    (ForkCondition::Block(0), OptimismSpecId::CONSTANTINOPLE),
    (ForkCondition::Block(0), OptimismSpecId::PETERSBURG),
    (ForkCondition::Block(0), OptimismSpecId::ISTANBUL),
    (ForkCondition::Block(0), OptimismSpecId::MUIR_GLACIER),
    (ForkCondition::Block(3_950_000), OptimismSpecId::BERLIN),
    (ForkCondition::Block(105_235_063), OptimismSpecId::LONDON),
    (
        ForkCondition::Block(105_235_063),
        OptimismSpecId::ARROW_GLACIER,
    ),
    (
        ForkCondition::Block(105_235_063),
        OptimismSpecId::GRAY_GLACIER,
    ),
    (ForkCondition::Block(105_235_063), OptimismSpecId::MERGE),
    (ForkCondition::Block(105_235_063), OptimismSpecId::BEDROCK),
    (ForkCondition::Block(105_235_063), OptimismSpecId::REGOLITH),
    (
        ForkCondition::Timestamp(1_704_992_401),
        OptimismSpecId::SHANGHAI,
    ),
    (
        ForkCondition::Timestamp(1_704_992_401),
        OptimismSpecId::CANYON,
    ),
    (
        ForkCondition::Timestamp(1_710_374_401),
        OptimismSpecId::CANCUN,
    ),
    (
        ForkCondition::Timestamp(1_710_374_401),
        OptimismSpecId::ECOTONE,
    ),
    (
        ForkCondition::Timestamp(1_720_627_201),
        OptimismSpecId::FJORD,
    ),
];

fn mainnet_config() -> &'static ChainConfig<OptimismSpecId> {
    static CONFIG: OnceLock<ChainConfig<OptimismSpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = MAINNET_HARDFORKS.into();

        ChainConfig {
            name: "mainnet".to_string(),
            hardfork_activations,
        }
    })
}

const SEPOLIA_HARDFORKS: &[(ForkCondition, OptimismSpecId)] = &[
    (ForkCondition::Block(0), OptimismSpecId::FRONTIER),
    (ForkCondition::Block(0), OptimismSpecId::HOMESTEAD),
    (ForkCondition::Block(0), OptimismSpecId::TANGERINE),
    (ForkCondition::Block(0), OptimismSpecId::SPURIOUS_DRAGON),
    (ForkCondition::Block(0), OptimismSpecId::BYZANTIUM),
    (ForkCondition::Block(0), OptimismSpecId::CONSTANTINOPLE),
    (ForkCondition::Block(0), OptimismSpecId::PETERSBURG),
    (ForkCondition::Block(0), OptimismSpecId::ISTANBUL),
    (ForkCondition::Block(0), OptimismSpecId::MUIR_GLACIER),
    (ForkCondition::Block(0), OptimismSpecId::BERLIN),
    (ForkCondition::Block(0), OptimismSpecId::LONDON),
    (ForkCondition::Block(0), OptimismSpecId::ARROW_GLACIER),
    (ForkCondition::Block(0), OptimismSpecId::GRAY_GLACIER),
    (ForkCondition::Block(0), OptimismSpecId::MERGE),
    (ForkCondition::Block(0), OptimismSpecId::BEDROCK),
    (ForkCondition::Block(0), OptimismSpecId::REGOLITH),
    (
        ForkCondition::Timestamp(1_699_981_200),
        OptimismSpecId::SHANGHAI,
    ),
    (
        ForkCondition::Timestamp(1_699_981_200),
        OptimismSpecId::CANYON,
    ),
    (
        ForkCondition::Timestamp(1_708_534_800),
        OptimismSpecId::CANCUN,
    ),
    (
        ForkCondition::Timestamp(1_708_534_800),
        OptimismSpecId::ECOTONE,
    ),
    (
        ForkCondition::Timestamp(1_716_998_400),
        OptimismSpecId::FJORD,
    ),
];

fn sepolia_config() -> &'static ChainConfig<OptimismSpecId> {
    static CONFIG: OnceLock<ChainConfig<OptimismSpecId>> = OnceLock::new();

    CONFIG.get_or_init(|| {
        let hardfork_activations = SEPOLIA_HARDFORKS.into();

        ChainConfig {
            name: "sepolia".to_string(),
            hardfork_activations,
        }
    })
}

// Source:
// <https://docs.optimism.io/builders/node-operators/network-upgrades>
fn chain_configs() -> &'static HashMap<u64, &'static ChainConfig<OptimismSpecId>> {
    static CONFIGS: OnceLock<HashMap<u64, &'static ChainConfig<OptimismSpecId>>> = OnceLock::new();

    CONFIGS.get_or_init(|| {
        let mut hardforks = HashMap::new();
        hardforks.insert(10, mainnet_config());
        hardforks.insert(11_155_420, sepolia_config());

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
pub fn chain_hardfork_activations(chain_id: u64) -> Option<&'static Activations<OptimismSpecId>> {
    chain_configs()
        .get(&chain_id)
        .map(|config| &config.hardfork_activations)
}
