use edr_eth::{eips::eip2930, Address, Blob, Bytes, B256, U256};

/// For specifying input to methods requiring a transaction object, like
/// `eth_call` and `eth_estimateGas`
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallRequest {
    /// the address from which the transaction should be sent
    pub from: Option<Address>,
    /// the address to which the transaction should be sent
    pub to: Option<Address>,
    /// gas
    #[serde(default, with = "alloy_serde::quantity::opt")]
    pub gas: Option<u64>,
    /// gas price
    pub gas_price: Option<U256>,
    /// max base fee per gas sender is willing to pay
    pub max_fee_per_gas: Option<U256>,
    /// miner tip
    pub max_priority_fee_per_gas: Option<U256>,
    /// transaction value
    pub value: Option<U256>,
    /// transaction data
    #[serde(alias = "input")]
    pub data: Option<Bytes>,
    /// warm storage access pre-payment
    pub access_list: Option<Vec<eip2930::AccessListItem>>,
    /// EIP-2718 type
    #[serde(default, rename = "type", with = "alloy_serde::quantity::opt")]
    pub transaction_type: Option<u8>,
    /// Blobs (EIP-4844)
    pub blobs: Option<Vec<Blob>>,
    /// Blob versioned hashes (EIP-4844)
    pub blob_hashes: Option<Vec<B256>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_alias() -> anyhow::Result<()> {
        const JSON_WITH_DATA: &str = r#"{
            "from":"0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266",
            "to":"0x5fbdb2315678afecb367f032d93f642f64180aa3",
            "data":"0x8b1329e0"
        }"#;

        const JSON_WITH_INPUT: &str = r#"{
            "from":"0x0000000000000000000000000000000000000000",
            "input":"0x8b1329e0",
            "to":"0x5fbdb2315678afecb367f032d93f642f64180aa3"
        }"#;

        const JSON_WITH_BOTH: &str = r#"{
            "from":"0x0000000000000000000000000000000000000000",
            "data":"0x8b1329e0",
            "input":"0x8b1329e0",
            "to":"0x5fbdb2315678afecb367f032d93f642f64180aa3"
        }"#;

        let with_data: CallRequest = serde_json::from_str(JSON_WITH_DATA)?;
        let with_input: CallRequest = serde_json::from_str(JSON_WITH_INPUT)?;
        assert_eq!(with_data.data, with_input.data);

        let error: serde_json::Error = serde_json::from_str::<CallRequest>(JSON_WITH_BOTH)
            .expect_err("Should fail due to duplicate fields");

        assert_eq!(
            error.to_string(),
            "duplicate field `data` at line 4 column 19"
        );

        Ok(())
    }
}
