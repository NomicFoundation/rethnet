use crate::U256;

/// Fee history for the returned block range. This can be a subsection of the
/// requested range if not all blocks are available.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct FeeHistoryResult {
    /// Lowest number block of returned range.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde::u64"))]
    pub oldest_block: u64,
    /// An array of block base fees per gas. This includes the next block after
    /// the newest of the returned range, because this value can be derived from
    /// the newest block. Zeroes are returned for pre-EIP-1559 blocks.
    pub base_fee_per_gas: Vec<U256>,
    /// An array of block gas used ratios. These are calculated as the ratio of
    /// gas used and gas limit.
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::serde::deserialize_float_vec")
    )]
    pub gas_used_ratio: Vec<f64>,
    /// A two-dimensional array of effective priority fees per gas at the
    /// requested block percentiles.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub reward: Option<Vec<Vec<U256>>>,
}

impl FeeHistoryResult {
    /// Constructs a new `FeeHistoryResult` with the oldest block and otherwise
    /// default fields.
    pub fn new(oldest_block: u64) -> Self {
        Self {
            oldest_block,
            base_fee_per_gas: Vec::default(),
            gas_used_ratio: Vec::default(),
            reward: Option::default(),
        }
    }
}
