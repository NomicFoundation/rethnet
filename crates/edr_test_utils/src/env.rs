//! Helper functions for environment variables

fn get_non_empty_env_var_or_panic(name: &'static str) -> String {
    let result = std::env::var_os(name)
        .unwrap_or_else(|| panic!("{name} environment variable not defined"))
        .into_string()
        .expect("Couldn't convert OsString into a String");
    if result.is_empty() {
        panic!("{name} environment variable is empty")
    } else {
        result
    }
}

/// Returns the Alchemy URL from the environment variables.
///
/// # Panics
///
/// Panics if the environment variable is not defined, or if it is empty.
pub fn get_alchemy_url() -> String {
    get_non_empty_env_var_or_panic("ALCHEMY_URL")
}

/// Returns the Infura URL from the environment variables.
///
/// # Panics
///
/// Panics if the environment variable is not defined, or if it is empty.
pub fn get_infura_url() -> String {
    get_non_empty_env_var_or_panic("INFURA_URL")
}

/// Enum representing the different types of networks.
pub enum NetworkType {
    Ethereum,
    Sepolia,
    Optimism,
    Arbitrum,
    Polygon,
}

/// Return the URL of a specific network Alchemy from environment variables.
///
/// # Panics
///
/// Panics if the environment variable is not defined, or if it is empty.
pub fn get_alchemy_url_for_network(network_type: NetworkType) -> String {
    let alchemy_url = get_alchemy_url();

    let url_without_network = alchemy_url
        .strip_prefix("https://eth-mainnet")
        .expect("Failed to remove alchemy url network prefix");
    match network_type {
        NetworkType::Ethereum => alchemy_url,
        NetworkType::Sepolia => format!("https://eth-sepolia{url_without_network}"),
        NetworkType::Optimism => format!("https://opt-mainnet{url_without_network}"),
        NetworkType::Arbitrum => format!("https://arb-mainnet{url_without_network}"),
        NetworkType::Polygon => format!("https://polygon-mainnet{url_without_network}"),
    }
}
