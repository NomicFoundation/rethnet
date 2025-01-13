#[macro_export]
macro_rules! impl_execution_receipt_tests {
    ($chain_spec:ty, $block_receipt_factory:expr => {
        $(
            $name:ident, $hardfork:expr => $receipt:expr,
        )+
    }) => {
        $(
            paste::item! {
                #[test]
                fn [<typed_receipt_rpc_receipt_roundtrip_ $name>]() -> anyhow::Result<()> {
                    use edr_eth::{
                        log::{FilterLog, FullBlockLog, ReceiptLog},
                        receipt::{MapReceiptLogs as _, ReceiptFactory as _, TransactionReceipt},
                        spec::ChainSpec,
                        Address, B256, U256,
                    };

                    use $crate::{RpcTypeFrom as _, spec::RpcSpec};

                    let block_hash = B256::random();
                    let block_number = 10u64;
                    let transaction_hash = B256::random();
                    let transaction_index = 5u64;

                    let execution_receipt = $receipt;

                    let mut log_index = 0;
                    let execution_receipt = execution_receipt.map_logs(|log| FilterLog {
                        inner: FullBlockLog {
                            inner: ReceiptLog {
                                inner: log,
                                transaction_hash,
                            },
                            block_hash,
                            block_number,
                            log_index: {
                                let index = log_index;
                                log_index += 1;
                                index
                            },
                            transaction_index,
                        },
                        removed: false,
                    });

                    let transaction_receipt = TransactionReceipt {
                        inner: execution_receipt,
                        transaction_hash,
                        transaction_index,
                        from: Address::random(),
                        to: Some(Address::random()),
                        contract_address: Some(Address::random()),
                        gas_used: 100,
                        effective_gas_price: Some(U256::from(100u64)),
                    };

                    // ASSUMPTION: The transaction data doesn't matter for this test, so we can use a default transaction.
                    let transaction = <$chain_spec as ChainSpec>::SignedTransaction::default();

                    let receipt_factory = $block_receipt_factory;
                    let block_receipt = receipt_factory.create_receipt($hardfork, &transaction, transaction_receipt, &block_hash, block_number);

                    let rpc_receipt = <$chain_spec as RpcSpec>::RpcReceipt::rpc_type_from(&block_receipt, Default::default());

                    let serialized = serde_json::to_string(&rpc_receipt)?;
                    let deserialized = serde_json::from_str(&serialized)?;
                    assert_eq!(rpc_receipt, deserialized);

                    // This is necessary to ensure that the deser implementation doesn't expect a
                    // &str where a String can be passed.
                    let serialized = serde_json::to_value(&rpc_receipt)?;
                    let deserialized = serde_json::from_value(serialized)?;
                    assert_eq!(rpc_receipt, deserialized);

                    let receipt = rpc_receipt.try_into()?;
                    assert_eq!(block_receipt, receipt);

                    Ok(())
                }
            }
        )+
    };
}
