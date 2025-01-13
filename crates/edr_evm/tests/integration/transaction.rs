#[cfg(feature = "test-remote")]
mod alchemy {
    macro_rules! impl_test_transaction_remote_hash {
        ($(
            $name:ident => $block_number:literal,
        )+) => {
            $(
                paste::item! {
                    #[tokio::test]
                    async fn [<transaction_remote_ $name _hash>]() {
                        use edr_eth::{
                            l1::L1ChainSpec,
                            transaction::{self, ExecutableTransaction as _},
                            PreEip1898BlockSpec,
                            B256
                        };
                        use edr_rpc_eth::client::EthRpcClient;
                        use edr_test_utils::env::get_alchemy_url;

                        let client = EthRpcClient::<L1ChainSpec>::new(&get_alchemy_url(), edr_defaults::CACHE_DIR.into(), None).expect("url ok");

                        let block = client
                            .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::Number($block_number))
                            .await
                            .expect("Should succeed");

                        let transaction_hashes: Vec<B256> = block
                            .transactions
                            .iter()
                            .map(|transaction| transaction.hash)
                            .collect();

                        let transactions =
                                block.transactions.into_iter().map(transaction::Signed::try_from).collect::<Result<Vec<_>, _>>()
                                    .expect("Conversion must succeed, as we're not retrieving a pending block");

                        for (index, transaction) in transactions.iter().enumerate() {
                            assert_eq!(transaction_hashes[index], *transaction.transaction_hash());
                        }
                    }
                }
            )+
        };
    }

    impl_test_transaction_remote_hash! {
        legacy => 1_500_000u64,
        eip155 => 2_675_000u64,
        eip2930 => 12_244_000u64,
        eip1559 => 12_965_000u64,
    }
}
