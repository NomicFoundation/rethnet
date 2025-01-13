use edr_eth::{
    filter::{LogFilterOptions, LogOutput, OneOrMore},
    l1::L1ChainSpec,
    Address, Blob, BlockSpec, BlockTag, Bytes, PreEip1898BlockSpec, B256, BYTES_PER_BLOB, U160,
    U256,
};
use edr_provider::{IntervalConfigRequest, MethodInvocation, Timestamp};
use edr_rpc_eth::{CallRequest, TransactionRequest};

use crate::common::{
    help_test_method_invocation_serde, help_test_method_invocation_serde_with_expected,
};

#[test]
fn test_serde_eth_accounts() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Accounts(()));
}

#[test]
fn test_serde_eth_block_number() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::BlockNumber(()));
}

#[test]
fn test_serde_eth_call() {
    let tx = CallRequest {
        from: Some(Address::from(U160::from(1))),
        to: Some(Address::from(U160::from(2))),
        gas: Some(3),
        gas_price: Some(U256::from(4)),
        max_fee_per_gas: None,
        max_priority_fee_per_gas: None,
        value: Some(U256::from(123568919)),
        data: Some(Bytes::from(&b"whatever"[..])),
        access_list: None,
        transaction_type: None,
        blobs: Some(vec![Blob::new([1u8; BYTES_PER_BLOB])]),
        blob_hashes: Some(vec![B256::from(U256::from(1))]),
    };
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Call(
        tx.clone(),
        Some(BlockSpec::latest()),
        None,
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::Call(tx.clone(), None, None),
        MethodInvocation::<L1ChainSpec>::Call(tx, Some(BlockSpec::latest()), None),
    );
}

#[test]
fn test_serde_eth_chain_id() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::ChainId(()));
}

#[test]
fn test_serde_eth_coinbase() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Coinbase(()));
}

#[test]
fn test_serde_eth_estimate_gas() {
    let tx = CallRequest {
        from: Some(Address::from(U160::from(1))),
        to: Some(Address::from(U160::from(2))),
        gas: Some(3),
        gas_price: Some(U256::from(4)),
        max_fee_per_gas: None,
        max_priority_fee_per_gas: None,
        value: Some(U256::from(123568919)),
        data: Some(Bytes::from(&b"whatever"[..])),
        access_list: None,
        transaction_type: None,
        blobs: None,
        blob_hashes: None,
    };
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EstimateGas(
        tx.clone(),
        Some(BlockSpec::latest()),
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::EstimateGas(tx.clone(), None),
        MethodInvocation::<L1ChainSpec>::EstimateGas(tx, Some(BlockSpec::pending())),
    );
}

#[test]
fn test_serde_eth_fee_history() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::FeeHistory(
        U256::from(3),
        BlockSpec::Number(100),
        Some(vec![0.5_f64, 10_f64, 80_f64, 90_f64, 99.5_f64]),
    ));
}

#[test]
fn test_serde_eth_gas_price() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GasPrice(()));
}

#[test]
fn test_serde_eth_get_balance() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetBalance(
        Address::from(U160::from(1)),
        Some(BlockSpec::latest()),
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::GetBalance(Address::from(U160::from(1)), None),
        MethodInvocation::<L1ChainSpec>::GetBalance(
            Address::from(U160::from(1)),
            Some(BlockSpec::latest()),
        ),
    );
}

#[test]
fn test_serde_eth_get_block_by_number() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetBlockByNumber(
        PreEip1898BlockSpec::Number(100),
        true,
    ));
}

#[test]
fn test_serde_eth_get_block_by_tag() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetBlockByNumber(
        PreEip1898BlockSpec::latest(),
        true,
    ));
}

#[test]
fn test_serde_eth_get_block_by_hash() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetBlockByHash(
        B256::from(U256::from(1)),
        true,
    ));
}

#[test]
fn test_serde_eth_get_transaction_count() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetTransactionCount(
        Address::from(U160::from(1)),
        Some(BlockSpec::latest()),
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::GetTransactionCount(Address::from(U160::from(1)), None),
        MethodInvocation::<L1ChainSpec>::GetTransactionCount(
            Address::from(U160::from(1)),
            Some(BlockSpec::latest()),
        ),
    );
}

#[test]
fn test_serde_eth_get_transaction() {
    help_test_method_invocation_serde(
        MethodInvocation::<L1ChainSpec>::GetBlockTransactionCountByHash(B256::from(U256::from(1))),
    );
}

#[test]
fn test_serde_eth_get_transaction_count_by_number() {
    help_test_method_invocation_serde(
        MethodInvocation::<L1ChainSpec>::GetBlockTransactionCountByNumber(
            PreEip1898BlockSpec::Number(100),
        ),
    );
}

#[test]
fn test_serde_eth_get_code() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetCode(
        Address::from(U160::from(1)),
        Some(BlockSpec::latest()),
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::GetCode(Address::from(U160::from(1)), None),
        MethodInvocation::<L1ChainSpec>::GetCode(
            Address::from(U160::from(1)),
            Some(BlockSpec::latest()),
        ),
    );
}

#[test]
fn test_serde_eth_get_filter_changes() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetFilterChanges(
        U256::from(100),
    ));
}

#[test]
fn test_serde_eth_get_filter_logs() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetFilterLogs(U256::from(
        100,
    )));
}

#[test]
fn test_serde_eth_get_logs_by_block_numbers() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetLogs(LogFilterOptions {
        from_block: Some(BlockSpec::Number(100)),
        to_block: Some(BlockSpec::Number(102)),
        block_hash: None,
        address: Some(OneOrMore::One(Address::from(U160::from(1)))),
        topics: None,
    }));
}

#[test]
fn test_serde_eth_get_logs_by_block_tags() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetLogs(LogFilterOptions {
        from_block: Some(BlockSpec::Tag(BlockTag::Safe)),
        to_block: Some(BlockSpec::latest()),
        block_hash: None,
        address: Some(OneOrMore::One(Address::from(U160::from(1)))),
        topics: Some(vec![Some(OneOrMore::One(B256::from(U256::from(1))))]),
    }));
}

#[test]
fn test_serde_eth_get_logs_by_block_hash() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetLogs(LogFilterOptions {
        from_block: None,
        to_block: None,
        block_hash: Some(B256::from(U256::from(1))),
        address: Some(OneOrMore::One(Address::from(U160::from(1)))),
        topics: Some(vec![Some(OneOrMore::One(B256::from(U256::from(1))))]),
    }));
}

#[test]
fn test_serde_eth_get_storage_at() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetStorageAt(
        Address::from(U160::from(1)),
        U256::ZERO,
        Some(BlockSpec::latest()),
    ));
    help_test_method_invocation_serde_with_expected(
        MethodInvocation::<L1ChainSpec>::GetStorageAt(
            Address::from(U160::from(1)),
            U256::ZERO,
            None,
        ),
        MethodInvocation::<L1ChainSpec>::GetStorageAt(
            Address::from(U160::from(1)),
            U256::ZERO,
            Some(BlockSpec::latest()),
        ),
    );
}

#[test]
fn test_serde_eth_get_tx_by_block_hash_and_index() {
    help_test_method_invocation_serde(
        MethodInvocation::<L1ChainSpec>::GetTransactionByBlockHashAndIndex(
            B256::from(U256::from(1)),
            U256::from(1),
        ),
    );
}

#[test]
fn test_serde_eth_get_tx_by_block_number_and_index() {
    help_test_method_invocation_serde(
        MethodInvocation::<L1ChainSpec>::GetTransactionByBlockNumberAndIndex(
            PreEip1898BlockSpec::Number(100),
            U256::from(1),
        ),
    );
}

#[test]
fn test_serde_eth_get_tx_by_hash() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetTransactionByHash(
        B256::from(U256::from(1)),
    ));
}

#[test]
fn test_serde_eth_get_tx_count_by_block_number() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetTransactionCount(
        Address::from(U160::from(1)),
        Some(BlockSpec::Number(100)),
    ));
}

#[test]
fn test_serde_eth_get_tx_count_by_block_tag() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetTransactionCount(
        Address::from(U160::from(1)),
        Some(BlockSpec::latest()),
    ));
}

#[test]
fn test_serde_eth_get_tx_receipt() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::GetTransactionReceipt(
        B256::from(U256::from(1)),
    ));
}

#[test]
fn test_serde_eth_mining() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Mining(()));
}

#[test]
fn test_serde_eth_new_block_filter() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::NewBlockFilter(()));
}

#[test]
fn test_serde_eth_new_filter() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::NewFilter(
        LogFilterOptions {
            from_block: Some(BlockSpec::Number(1000)),
            to_block: Some(BlockSpec::latest()),
            block_hash: None,
            address: Some(OneOrMore::One(Address::from(U160::from(1)))),
            topics: Some(vec![Some(OneOrMore::One(B256::from(U256::from(1))))]),
        },
    ));
}

#[test]
fn test_serde_eth_new_pending_transaction_filter() {
    help_test_method_invocation_serde(
        MethodInvocation::<L1ChainSpec>::NewPendingTransactionFilter(()),
    );
}

#[test]
fn test_serde_eth_pending_transactions() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::PendingTransactions(()));
}

#[test]
fn test_serde_eth_send_raw_transaction() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::SendRawTransaction(
        Bytes::from(&b"whatever"[..]),
    ));
}

#[test]
fn test_serde_eth_send_transaction() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::SendTransaction(
        TransactionRequest {
            from: Address::from(U160::from(1)),
            to: Some(Address::from(U160::from(2))),
            gas: Some(3_u64),
            gas_price: Some(U256::from(4)),
            max_fee_per_gas: None,
            value: Some(U256::from(123568919)),
            data: Some(Bytes::from(&b"whatever"[..])),
            nonce: None,
            chain_id: None,
            access_list: None,
            max_priority_fee_per_gas: None,
            transaction_type: None,
            blobs: Some(vec![Blob::new([1u8; BYTES_PER_BLOB])]),
            blob_hashes: Some(vec![B256::from(U256::from(1))]),
        },
    ));
}

#[test]
fn test_serde_personal_sign() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::PersonalSign(
        Bytes::from(&b"whatever"[..]),
        Address::from(U160::from(1)),
    ));
}

#[test]
fn test_serde_eth_sign() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EthSign(
        Address::from(U160::from(1)),
        Bytes::from(&b"whatever"[..]),
    ));
}

macro_rules! impl_serde_eth_subscribe_tests {
    ($(
        $name:ident => $variant:expr,
    )+) => {
        $(
            paste::item! {
                #[test]
                fn [<test_serde_eth_subscribe_ $name _without_filter>]() {
                    use edr_eth::filter::SubscriptionType;

                    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Subscribe($variant, None));
                }

                #[test]
                fn [<test_serde_eth_subscribe_ $name _with_filter>]() {
                    use edr_eth::filter::SubscriptionType;

                    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Subscribe($variant, Some(LogFilterOptions {
                        from_block: Some(BlockSpec::Number(1000)),
                        to_block: Some(BlockSpec::latest()),
                        block_hash: None,
                        address: Some(OneOrMore::One(Address::from(U160::from(1)))),
                        topics: Some(vec![Some(OneOrMore::One(B256::from(U256::from(1))))]),
                    })));
                }
            }
        )+
    };
}

impl_serde_eth_subscribe_tests! {
    logs => SubscriptionType::Logs,
    new_pending_transactions => SubscriptionType::NewPendingTransactions,
    new_heads => SubscriptionType::NewHeads,
}

#[test]
fn test_serde_eth_syncing() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Syncing(()));
}

#[test]
fn test_serde_eth_uninstall_filter() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::UninstallFilter(
        U256::from(100),
    ));
}

#[test]
fn test_serde_eth_unsubscribe() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Unsubscribe(U256::from(
        100,
    )));
}

fn help_test_serde_value<T>(value: T)
where
    T: PartialEq + std::fmt::Debug + serde::de::DeserializeOwned + serde::Serialize,
{
    let serialized = serde_json::json!(value).to_string();

    let deserialized: T = serde_json::from_str(&serialized)
        .unwrap_or_else(|_| panic!("should have successfully deserialized json {serialized}"));

    assert_eq!(value, deserialized);
}

#[test]
fn test_serde_log_output() {
    help_test_serde_value(LogOutput {
        removed: false,
        log_index: Some(0),
        transaction_index: Some(99),
        transaction_hash: Some(B256::from(U256::from(1))),
        block_hash: Some(B256::from(U256::from(2))),
        block_number: Some(0),
        address: Address::from(U160::from(1)),
        data: Bytes::from_static(b"whatever"),
        topics: vec![B256::from(U256::from(3)), B256::from(U256::from(3))],
    });
}

#[test]
fn test_serde_one_or_more_addresses() {
    help_test_serde_value(OneOrMore::One(Address::from(U160::from(1))));
    help_test_serde_value(OneOrMore::Many(vec![
        Address::from(U160::from(1)),
        Address::from(U160::from(1)),
    ]));
}

#[test]
fn test_evm_increase_time() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmIncreaseTime(
        Timestamp::from(12345),
    ));
}

#[test]
fn test_evm_mine() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmMine(Some(
        Timestamp::from(12345),
    )));
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmMine(None));
}

#[test]
fn test_evm_set_next_block_timestamp() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmSetNextBlockTimestamp(
        Timestamp::from(12345),
    ));
}

#[test]
fn test_serde_web3_client_version() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Web3ClientVersion(()));
}

#[test]
fn test_serde_web3_sha3() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::Web3Sha3(Bytes::from(
        &b"whatever"[..],
    )));
}

#[test]
fn test_evm_set_automine() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmSetAutomine(false));
}

#[test]
fn test_evm_set_interval_mining() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmSetIntervalMining(
        IntervalConfigRequest::FixedOrDisabled(1000),
    ));
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmSetIntervalMining(
        IntervalConfigRequest::Range([1000, 5000]),
    ));
}

#[test]
fn test_evm_snapshot() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::EvmSnapshot(()));
}

#[test]
fn test_net_listening() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::NetListening(()));
}

#[test]
fn test_net_peer_count() {
    help_test_method_invocation_serde(MethodInvocation::<L1ChainSpec>::NetPeerCount(()));
}

#[test]
fn test_personal_sign() {
    let call = MethodInvocation::<L1ChainSpec>::PersonalSign(
        Bytes::from(&b"whatever"[..]),
        Address::from(U160::from(1)),
    );

    let serialized = serde_json::json!(call).to_string();

    let call_deserialized: MethodInvocation<L1ChainSpec> = serde_json::from_str(&serialized)
        .unwrap_or_else(|_| panic!("should have successfully deserialized json {serialized}"));

    assert_eq!(call, call_deserialized);
}

#[test]
fn test_eth_sign() {
    let call = MethodInvocation::<L1ChainSpec>::EthSign(
        Address::from(U160::from(1)),
        Bytes::from(&b"whatever"[..]),
    );

    let serialized = serde_json::json!(call).to_string();

    let call_deserialized: MethodInvocation<L1ChainSpec> = serde_json::from_str(&serialized)
        .unwrap_or_else(|_| panic!("should have successfully deserialized json {serialized}"));

    assert_eq!(call, call_deserialized);
}
