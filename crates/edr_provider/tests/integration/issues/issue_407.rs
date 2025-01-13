use std::sync::Arc;

use edr_eth::l1::L1ChainSpec;
use edr_provider::{test_utils::create_test_config, time::CurrentTime, NoopLogger, Provider};
use edr_solidity::contract_decoder::ContractDecoder;
use serde_json::json;
use tokio::runtime;

// https://github.com/NomicFoundation/edr/issues/407

#[tokio::test(flavor = "multi_thread")]
async fn issue_407_uint() -> anyhow::Result<()> {
    let config = create_test_config();
    let logger = Box::new(NoopLogger::<L1ChainSpec>::default());
    let subscriber = Box::new(|_event| {});
    let provider = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        config,
        Arc::<ContractDecoder>::default(),
        CurrentTime,
    )?;

    let request_with_uint = json!({
      "method": "eth_signTypedData_v4",
      "params": [
        "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266",
        {
          "types": {
            "Test": [
              {
                "name": "amount",
                "type": "uint256"
              }
            ],
            "EIP712Domain": [
              {
                "name": "name",
                "type": "string"
              },
              {
                "name": "version",
                "type": "string"
              },
              {
                "name": "chainId",
                "type": "uint256"
              },
              {
                "name": "verifyingContract",
                "type": "address"
              }
            ]
          },
          "domain": {
            "name": "TestName",
            "version": "1",
            "chainId": "0x7a69",
            "verifyingContract": "0x1111111111111111111111111111111111111111"
          },
          "primaryType": "Test",
          "message": {
            "amount": "1234"
          }
        }
      ]
    });

    let response = provider.handle_request(serde_json::from_value(request_with_uint)?)?;

    assert_eq!(response.result, "0x2d9b08a9086931cc3ebb9ae446d440e43f0e4ca0abedd2d973af8278c5471bb54181a8dae6018d14d29d62facc535fbba5b4010cdb3f06c0ddcf72e2663583361b");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn issue_407_int() -> anyhow::Result<()> {
    let config = create_test_config();
    let logger = Box::new(NoopLogger::<L1ChainSpec>::default());
    let subscriber = Box::new(|_event| {});
    let provider = Provider::new(
        runtime::Handle::current(),
        logger,
        subscriber,
        config,
        Arc::<ContractDecoder>::default(),
        CurrentTime,
    )?;

    let request_with_uint = json!({
      "method": "eth_signTypedData_v4",
      "params": [
        "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266",
        {
          "types": {
            "Test": [
              {
                "name": "amount",
                "type": "int256"
              }
            ],
            "EIP712Domain": [
              {
                "name": "name",
                "type": "string"
              },
              {
                "name": "version",
                "type": "string"
              },
              {
                "name": "chainId",
                "type": "uint256"
              },
              {
                "name": "verifyingContract",
                "type": "address"
              }
            ]
          },
          "domain": {
            "name": "TestName",
            "version": "1",
            "chainId": "0x7a69",
            "verifyingContract": "0x1111111111111111111111111111111111111111"
          },
          "primaryType": "Test",
          "message": {
            "amount": "1234"
          }
        }
      ]
    });

    let response = provider.handle_request(serde_json::from_value(request_with_uint)?)?;

    assert_eq!(response.result, "0x30622c2e4318a3ffb2755ca111f42409bd5a4190b4b8a9b5f42227313708ecb54889d229dfb7dcfb246f15e7567ef7471fb26a7e99d83631d266a144502ee29f1c");

    Ok(())
}
