use std::{fmt::Debug, path::PathBuf};

use derive_where::derive_where;
use edr_eth::{
    account::{AccountInfo, KECCAK_EMPTY},
    fee_history::FeeHistoryResult,
    filter::{LogFilterOptions, OneOrMore},
    log::FilterLog,
    reward_percentile::RewardPercentile,
    Address, BlockSpec, Bytecode, Bytes, PreEip1898BlockSpec, B256, U256, U64,
};
use edr_rpc_client::RpcClient;
pub use edr_rpc_client::{header, HeaderMap, RpcClientError};
use futures::StreamExt;

use crate::{
    fork::ForkMetadata,
    request_methods::RequestMethod,
    spec::{GetBlockNumber, RpcSpec},
};

// Constrain parallel requests to avoid rate limiting on transport level and
// thundering herd during backoff.
const MAX_PARALLEL_REQUESTS: usize = 20;

#[derive_where(Debug)]
pub struct EthRpcClient<RpcSpecT: RpcSpec> {
    inner: RpcClient<RequestMethod>,
    phantom: std::marker::PhantomData<RpcSpecT>,
}

impl<RpcSpecT: RpcSpec> EthRpcClient<RpcSpecT> {
    /// Creates a new instance, given a remote node URL.
    ///
    /// The cache directory is the global EDR cache directory configured by the
    /// user.
    pub fn new(
        url: &str,
        cache_dir: PathBuf,
        extra_headers: Option<HeaderMap>,
    ) -> Result<Self, RpcClientError> {
        let inner = RpcClient::new(url, cache_dir, extra_headers)?;
        Ok(Self {
            inner,
            phantom: std::marker::PhantomData,
        })
    }

    /// Calls `eth_blockNumber` and returns the block number.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn block_number(&self) -> Result<u64, RpcClientError> {
        self.inner.block_number().await
    }

    /// Calls `eth_chainId` and returns the chain ID.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn chain_id(&self) -> Result<u64, RpcClientError> {
        self.inner.chain_id().await
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    /// Calls `eth_feeHistory` and returns the fee history.
    pub async fn fee_history(
        &self,
        block_count: u64,
        newest_block: BlockSpec,
        reward_percentiles: Option<Vec<RewardPercentile>>,
    ) -> Result<FeeHistoryResult, RpcClientError> {
        self.inner
            .call(RequestMethod::FeeHistory(
                U256::from(block_count),
                newest_block,
                reward_percentiles,
            ))
            .await
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    /// Fetches the latest block number, chain ID, and network ID concurrently.
    pub async fn fork_metadata(&self) -> Result<ForkMetadata, RpcClientError> {
        let network_id = self.network_id();
        let block_number = self.inner.block_number();
        let chain_id = self.inner.chain_id();

        let (network_id, block_number, chain_id) =
            tokio::try_join!(network_id, block_number, chain_id)?;

        Ok(ForkMetadata {
            chain_id,
            network_id,
            latest_block_number: block_number,
        })
    }

    /// Submits three concurrent RPC method invocations in order to obtain
    /// the set of data contained in [`AccountInfo`].
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_account_info(
        &self,
        address: Address,
        block: Option<BlockSpec>,
    ) -> Result<AccountInfo, RpcClientError> {
        let balance = self.get_balance(address, block.clone());
        let nonce = self.get_transaction_count(address, block.clone());
        let code = self.get_code(address, block.clone());

        let (balance, nonce, code) = tokio::try_join!(balance, nonce, code)?;

        let code = if code.is_empty() {
            None
        } else {
            Some(Bytecode::new_raw(code))
        };

        Ok(AccountInfo {
            balance,
            code_hash: code.as_ref().map_or(KECCAK_EMPTY, Bytecode::hash_slow),
            code,
            nonce: nonce.to(),
        })
    }

    /// Fetches account infos for multiple addresses using concurrent requests.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_account_infos(
        &self,
        addresses: &[Address],
        block: Option<BlockSpec>,
    ) -> Result<Vec<AccountInfo>, RpcClientError> {
        futures::stream::iter(addresses.iter())
            .map(|address| self.get_account_info(*address, block.clone()))
            .buffered(MAX_PARALLEL_REQUESTS / 3 + 1)
            .collect::<Vec<Result<AccountInfo, RpcClientError>>>()
            .await
            .into_iter()
            .collect()
    }

    /// Calls `eth_getBlockByHash` and returns the transaction's hash.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_block_by_hash(
        &self,
        hash: B256,
    ) -> Result<Option<RpcSpecT::RpcBlock<B256>>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetBlockByHash(hash, false))
            .await
    }

    /// Calls `eth_getBalance`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_balance(
        &self,
        address: Address,
        block: Option<BlockSpec>,
    ) -> Result<U256, RpcClientError> {
        self.inner
            .call(RequestMethod::GetBalance(address, block))
            .await
    }

    /// Calls `eth_getBlockByHash` and returns the transaction's data.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_block_by_hash_with_transaction_data(
        &self,
        hash: B256,
    ) -> Result<Option<RpcSpecT::RpcBlock<RpcSpecT::RpcTransaction>>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetBlockByHash(hash, true))
            .await
    }

    /// Calls `eth_getBlockByNumber` and returns the transaction's hash.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_block_by_number(
        &self,
        spec: PreEip1898BlockSpec,
    ) -> Result<Option<RpcSpecT::RpcBlock<B256>>, RpcClientError> {
        self.inner
            .call_with_resolver(
                RequestMethod::GetBlockByNumber(spec, false),
                |block: &Option<RpcSpecT::RpcBlock<B256>>| {
                    block.as_ref().and_then(GetBlockNumber::number)
                },
            )
            .await
    }

    /// Calls `eth_getBlockByNumber` and returns the transaction's data.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_block_by_number_with_transaction_data(
        &self,
        spec: PreEip1898BlockSpec,
    ) -> Result<RpcSpecT::RpcBlock<RpcSpecT::RpcTransaction>, RpcClientError> {
        self.inner
            .call_with_resolver(
                RequestMethod::GetBlockByNumber(spec, true),
                |block: &RpcSpecT::RpcBlock<RpcSpecT::RpcTransaction>| block.number(),
            )
            .await
    }

    /// Calls `eth_getCode`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_code(
        &self,
        address: Address,
        block: Option<BlockSpec>,
    ) -> Result<Bytes, RpcClientError> {
        self.inner
            .call(RequestMethod::GetCode(address, block))
            .await
    }

    /// Calls `eth_getLogs` using a starting and ending block (inclusive).
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_logs_by_range(
        &self,
        from_block: BlockSpec,
        to_block: BlockSpec,
        address: Option<OneOrMore<Address>>,
        topics: Option<Vec<Option<OneOrMore<B256>>>>,
    ) -> Result<Vec<FilterLog>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetLogs(LogFilterOptions {
                from_block: Some(from_block),
                to_block: Some(to_block),
                block_hash: None,
                address,
                topics,
            }))
            .await
    }

    /// Calls `eth_getTransactionByHash`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_transaction_by_hash(
        &self,
        tx_hash: B256,
    ) -> Result<Option<RpcSpecT::RpcTransaction>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetTransactionByHash(tx_hash))
            .await
    }

    /// Calls `eth_getTransactionCount`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_transaction_count(
        &self,
        address: Address,
        block: Option<BlockSpec>,
    ) -> Result<U256, RpcClientError> {
        self.inner
            .call(RequestMethod::GetTransactionCount(address, block))
            .await
    }

    /// Calls `eth_getTransactionReceipt`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_transaction_receipt(
        &self,
        tx_hash: B256,
    ) -> Result<Option<RpcSpecT::RpcReceipt>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetTransactionReceipt(tx_hash))
            .await
    }

    /// Methods for retrieving multiple transaction receipts using concurrent
    /// requests.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_transaction_receipts(
        &self,
        hashes: impl IntoIterator<Item = &B256> + Debug,
    ) -> Result<Option<Vec<RpcSpecT::RpcReceipt>>, RpcClientError> {
        let requests = hashes
            .into_iter()
            .map(|transaction_hash| self.get_transaction_receipt(*transaction_hash))
            .collect::<Vec<_>>();

        futures::stream::iter(requests)
            .buffered(MAX_PARALLEL_REQUESTS)
            .collect::<Vec<Result<Option<RpcSpecT::RpcReceipt>, RpcClientError>>>()
            .await
            .into_iter()
            .collect()
    }

    /// Calls `eth_getStorageAt`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn get_storage_at(
        &self,
        address: Address,
        position: U256,
        block: Option<BlockSpec>,
    ) -> Result<Option<U256>, RpcClientError> {
        self.inner
            .call(RequestMethod::GetStorageAt(address, position, block))
            .await
    }

    /// Whether the block number should be cached based on its depth.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn is_cacheable_block_number(
        &self,
        block_number: u64,
    ) -> Result<bool, RpcClientError> {
        self.inner.is_cacheable_block_number(block_number).await
    }

    /// Calls `net_version`.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub async fn network_id(&self) -> Result<u64, RpcClientError> {
        self.inner
            .call::<U64>(RequestMethod::NetVersion(()))
            .await
            .map(|network_id| network_id.as_limbs()[0])
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Deref, str::FromStr};

    use edr_eth::l1::L1ChainSpec;
    use reqwest::StatusCode;
    use tempfile::TempDir;

    use super::*;

    struct TestRpcClient {
        client: EthRpcClient<L1ChainSpec>,

        // Need to keep the tempdir around to prevent it from being deleted
        // Only accessed when feature = "test-remote", hence the allow.
        #[allow(dead_code)]
        cache_dir: TempDir,
    }

    impl TestRpcClient {
        fn new(url: &str) -> Self {
            let tempdir = TempDir::new().unwrap();
            Self {
                client: EthRpcClient::new(url, tempdir.path().into(), None).expect("url ok"),
                cache_dir: tempdir,
            }
        }
    }

    impl Deref for TestRpcClient {
        type Target = EthRpcClient<L1ChainSpec>;

        fn deref(&self) -> &Self::Target {
            &self.client
        }
    }

    #[tokio::test]
    async fn send_request_body_400_status() {
        const STATUS_CODE: u16 = 400;

        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/")
            .with_status(STATUS_CODE.into())
            .with_header("content-type", "text/plain")
            .create_async()
            .await;

        let hash =
            B256::from_str("0xc008e9f9bb92057dd0035496fbf4fb54f66b4b18b370928e46d6603933022222")
                .expect("failed to parse hash from string");

        let error = TestRpcClient::new(&server.url())
            .get_transaction_by_hash(hash)
            .await
            .expect_err("should have failed to due to a HTTP status error");

        if let RpcClientError::HttpStatus(error) = error {
            assert_eq!(
                reqwest::Error::from(error).status(),
                Some(StatusCode::from_u16(STATUS_CODE).unwrap())
            );
        } else {
            unreachable!("Invalid error: {error}");
        }

        mock.assert_async().await;
    }

    #[cfg(feature = "test-remote")]
    mod alchemy {
        use std::{fs::File, path::PathBuf};

        use edr_eth::{filter::OneOrMore, Address, BlockSpec, Bytes, PreEip1898BlockSpec, U256};
        use edr_test_utils::env::get_alchemy_url;
        use walkdir::WalkDir;

        use super::*;

        // The maximum block number that Alchemy allows
        const MAX_BLOCK_NUMBER: u64 = u64::MAX >> 1;

        impl TestRpcClient {
            fn files_in_cache(&self) -> Vec<PathBuf> {
                let mut files = Vec::new();
                for entry in WalkDir::new(&self.cache_dir)
                    .follow_links(true)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    if entry.file_type().is_file() {
                        files.push(entry.path().to_owned());
                    }
                }
                files
            }
        }

        #[tokio::test]
        async fn get_account_info_unknown_block() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let error = TestRpcClient::new(&alchemy_url)
                .get_account_info(dai_address, Some(BlockSpec::Number(MAX_BLOCK_NUMBER)))
                .await
                .expect_err("should have failed");

            if let RpcClientError::JsonRpcError { error, .. } = error {
                assert_eq!(error.code, -32602);
                assert_eq!(error.message, "Unknown block number");
                assert!(error.data.is_none());
            } else {
                unreachable!("Invalid error: {error}");
            }
        }

        #[tokio::test]
        async fn get_account_infos() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");
            let hardhat_default_address =
                Address::from_str("0xbe862ad9abfe6f22bcb087716c7d89a26051f74c")
                    .expect("failed to parse address");

            let account_infos = TestRpcClient::new(&alchemy_url)
                .get_account_infos(
                    &[dai_address, hardhat_default_address],
                    Some(BlockSpec::latest()),
                )
                .await
                .expect("should have succeeded");

            assert_eq!(account_infos.len(), 2);
        }

        #[tokio::test]
        async fn get_block_by_hash_some() {
            let alchemy_url = get_alchemy_url();

            let hash = B256::from_str(
                "0x71d5e7c8ff9ea737034c16e333a75575a4a94d29482e0c2b88f0a6a8369c1812",
            )
            .expect("failed to parse hash from string");

            let block = TestRpcClient::new(&alchemy_url)
                .get_block_by_hash(hash)
                .await
                .expect("should have succeeded");

            assert!(block.is_some());
            let block = block.unwrap();

            assert_eq!(block.hash, Some(hash));
            assert_eq!(block.transactions.len(), 192);
        }

        #[tokio::test]
        async fn get_block_by_hash_with_transaction_data_some() {
            let alchemy_url = get_alchemy_url();

            let hash = B256::from_str(
                "0x71d5e7c8ff9ea737034c16e333a75575a4a94d29482e0c2b88f0a6a8369c1812",
            )
            .expect("failed to parse hash from string");

            let block = TestRpcClient::new(&alchemy_url)
                .get_block_by_hash_with_transaction_data(hash)
                .await
                .expect("should have succeeded");

            assert!(block.is_some());
            let block = block.unwrap();

            assert_eq!(block.hash, Some(hash));
            assert_eq!(block.transactions.len(), 192);
        }

        #[tokio::test]
        async fn get_block_by_number_finalized_resolves() {
            let alchemy_url = get_alchemy_url();
            let client = TestRpcClient::new(&alchemy_url);

            assert_eq!(client.files_in_cache().len(), 0);

            client
                .get_block_by_number(PreEip1898BlockSpec::finalized())
                .await
                .expect("should have succeeded");

            // Finalized tag should be resolved and stored in cache.
            assert_eq!(client.files_in_cache().len(), 1);
        }

        #[tokio::test]
        async fn get_block_by_number_some() {
            let alchemy_url = get_alchemy_url();

            let block_number = 16222385;

            let block = TestRpcClient::new(&alchemy_url)
                .get_block_by_number(PreEip1898BlockSpec::Number(block_number))
                .await
                .expect("should have succeeded")
                .expect("Block must exist");

            assert_eq!(block.number, Some(block_number));
            assert_eq!(block.transactions.len(), 102);
        }

        #[tokio::test]
        async fn get_block_with_transaction_data_cached() {
            let alchemy_url = get_alchemy_url();
            let client = TestRpcClient::new(&alchemy_url);

            let block_spec = PreEip1898BlockSpec::Number(16220843);

            assert_eq!(client.files_in_cache().len(), 0);

            let block_from_remote = client
                .get_block_by_number_with_transaction_data(block_spec.clone())
                .await
                .expect("should have from remote");

            assert_eq!(client.files_in_cache().len(), 1);

            let block_from_cache = client
                .get_block_by_number_with_transaction_data(block_spec.clone())
                .await
                .expect("should have from remote");

            assert_eq!(block_from_remote, block_from_cache);
        }

        #[tokio::test]
        async fn get_earliest_block_with_transaction_data_resolves() {
            let alchemy_url = get_alchemy_url();
            let client = TestRpcClient::new(&alchemy_url);

            assert_eq!(client.files_in_cache().len(), 0);

            client
                .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::earliest())
                .await
                .expect("should have succeeded");

            // Earliest tag should be resolved to block number and it should be cached.
            assert_eq!(client.files_in_cache().len(), 1);
        }

        #[tokio::test]
        async fn get_latest_block() {
            let alchemy_url = get_alchemy_url();

            let _block = TestRpcClient::new(&alchemy_url)
                .get_block_by_number(PreEip1898BlockSpec::latest())
                .await
                .expect("should have succeeded");
        }

        #[tokio::test]
        async fn get_latest_block_with_transaction_data() {
            let alchemy_url = get_alchemy_url();

            let _block = TestRpcClient::new(&alchemy_url)
                .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::latest())
                .await
                .expect("should have succeeded");
        }

        #[tokio::test]
        async fn get_pending_block() {
            let alchemy_url = get_alchemy_url();

            let _block = TestRpcClient::new(&alchemy_url)
                .get_block_by_number(PreEip1898BlockSpec::pending())
                .await
                .expect("should have succeeded");
        }

        #[tokio::test]
        async fn get_pending_block_with_transaction_data() {
            let alchemy_url = get_alchemy_url();

            let _block = TestRpcClient::new(&alchemy_url)
                .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::pending())
                .await
                .expect("should have succeeded");
        }

        #[tokio::test]
        async fn get_logs_some() {
            let alchemy_url = get_alchemy_url();
            let logs = TestRpcClient::new(&alchemy_url)
                .get_logs_by_range(
                    BlockSpec::Number(10496585),
                    BlockSpec::Number(10496585),
                    Some(OneOrMore::One(
                        Address::from_str("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
                            .expect("failed to parse data"),
                    )),
                    None,
                )
                .await
                .expect("failed to get logs");

            assert_eq!(logs.len(), 12);
            // TODO: assert more things about the log(s)
            // TODO: consider asserting something about the logs bloom
        }

        #[tokio::test]
        async fn get_logs_future_from_block() {
            let alchemy_url = get_alchemy_url();
            let error = TestRpcClient::new(&alchemy_url)
                .get_logs_by_range(
                    BlockSpec::Number(MAX_BLOCK_NUMBER),
                    BlockSpec::Number(MAX_BLOCK_NUMBER),
                    Some(OneOrMore::One(
                        Address::from_str("0xffffffffffffffffffffffffffffffffffffffff")
                            .expect("failed to parse data"),
                    )),
                    None,
                )
                .await
                .expect_err("should have failed to get logs");

            if let RpcClientError::JsonRpcError { error, .. } = error {
                assert_eq!(error.code, -32000);
                assert_eq!(error.message, "One of the blocks specified in filter (fromBlock, toBlock or blockHash) cannot be found.");
                assert!(error.data.is_none());
            } else {
                unreachable!("Invalid error: {error}");
            }
        }

        #[tokio::test]
        async fn get_logs_future_to_block() {
            let alchemy_url = get_alchemy_url();
            let logs = TestRpcClient::new(&alchemy_url)
                .get_logs_by_range(
                    BlockSpec::Number(10496585),
                    BlockSpec::Number(MAX_BLOCK_NUMBER),
                    Some(OneOrMore::One(
                        Address::from_str("0xffffffffffffffffffffffffffffffffffffffff")
                            .expect("failed to parse data"),
                    )),
                    None,
                )
                .await
                .expect("should have succeeded");

            assert_eq!(logs, []);
        }

        #[tokio::test]
        async fn get_transaction_by_hash_some() {
            let alchemy_url = get_alchemy_url();

            let hash = B256::from_str(
                "0xc008e9f9bb92057dd0035496fbf4fb54f66b4b18b370928e46d6603933054d5a",
            )
            .expect("failed to parse hash from string");

            let tx = TestRpcClient::new(&alchemy_url)
                .get_transaction_by_hash(hash)
                .await
                .expect("failed to get transaction by hash");

            assert!(tx.is_some());
            let tx = tx.unwrap();

            assert_eq!(
                tx.block_hash,
                Some(
                    B256::from_str(
                        "0x88fadbb673928c61b9ede3694ae0589ac77ae38ec90a24a6e12e83f42f18c7e8"
                    )
                    .expect("couldn't parse data")
                )
            );
            assert_eq!(
                tx.block_number,
                Some(u64::from_str_radix("a74fde", 16).expect("couldn't parse data"))
            );
            assert_eq!(tx.hash, hash);
            assert_eq!(
                tx.from,
                Address::from_str("0x7d97fcdb98632a91be79d3122b4eb99c0c4223ee")
                    .expect("couldn't parse data")
            );
            assert_eq!(
                tx.gas,
                U256::from_str_radix("30d40", 16).expect("couldn't parse data")
            );
            assert_eq!(
                tx.gas_price,
                U256::from_str_radix("1e449a99b8", 16).expect("couldn't parse data")
            );
            assert_eq!(
            tx.input,
            Bytes::from(hex::decode("a9059cbb000000000000000000000000e2c1e729e05f34c07d80083982ccd9154045dcc600000000000000000000000000000000000000000000000000000004a817c800").unwrap())
        );
            assert_eq!(
                tx.nonce,
                u64::from_str_radix("653b", 16).expect("couldn't parse data")
            );
            assert_eq!(
                tx.r,
                U256::from_str_radix(
                    "eb56df45bd355e182fba854506bc73737df275af5a323d30f98db13fdf44393a",
                    16
                )
                .expect("couldn't parse data")
            );
            assert_eq!(
                tx.s,
                U256::from_str_radix(
                    "2c6efcd210cdc7b3d3191360f796ca84cab25a52ed8f72efff1652adaabc1c83",
                    16
                )
                .expect("couldn't parse data")
            );
            assert_eq!(
                tx.to,
                Some(
                    Address::from_str("dac17f958d2ee523a2206206994597c13d831ec7")
                        .expect("couldn't parse data")
                )
            );
            assert_eq!(
                tx.transaction_index,
                Some(u64::from_str_radix("88", 16).expect("couldn't parse data"))
            );
            assert_eq!(
                tx.v,
                u64::from_str_radix("1c", 16).expect("couldn't parse data")
            );
            assert_eq!(
                tx.value,
                U256::from_str_radix("0", 16).expect("couldn't parse data")
            );
        }

        #[tokio::test]
        async fn get_transaction_count_some() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let transaction_count = TestRpcClient::new(&alchemy_url)
                .get_transaction_count(dai_address, Some(BlockSpec::Number(16220843)))
                .await
                .expect("should have succeeded");

            assert_eq!(transaction_count, U256::from(1));
        }

        #[tokio::test]
        async fn get_transaction_count_future_block() {
            let alchemy_url = get_alchemy_url();

            let missing_address = Address::from_str("0xffffffffffffffffffffffffffffffffffffffff")
                .expect("failed to parse address");

            let error = TestRpcClient::new(&alchemy_url)
                .get_transaction_count(missing_address, Some(BlockSpec::Number(MAX_BLOCK_NUMBER)))
                .await
                .expect_err("should have failed");

            if let RpcClientError::JsonRpcError { error, .. } = error {
                assert_eq!(error.code, -32602);
                assert_eq!(error.message, "Unknown block number");
                assert!(error.data.is_none());
            } else {
                unreachable!("Invalid error: {error}");
            }
        }

        #[tokio::test]
        async fn get_transaction_receipt_some() {
            let alchemy_url = get_alchemy_url();

            let hash = B256::from_str(
                "0xc008e9f9bb92057dd0035496fbf4fb54f66b4b18b370928e46d6603933054d5a",
            )
            .expect("failed to parse hash from string");

            let receipt = TestRpcClient::new(&alchemy_url)
                .get_transaction_receipt(hash)
                .await
                .expect("failed to get transaction by hash");

            assert!(receipt.is_some());
            let receipt = receipt.unwrap();

            assert_eq!(
                receipt.block_hash,
                B256::from_str(
                    "0x88fadbb673928c61b9ede3694ae0589ac77ae38ec90a24a6e12e83f42f18c7e8"
                )
                .expect("couldn't parse data")
            );
            assert_eq!(receipt.block_number, 0xa74fde);
            assert_eq!(receipt.contract_address, None);
            assert_eq!(receipt.cumulative_gas_used, 0x56c81b);
            assert_eq!(
                receipt.effective_gas_price,
                Some(U256::from_str_radix("1e449a99b8", 16).expect("couldn't parse data"))
            );
            assert_eq!(
                receipt.from,
                Address::from_str("0x7d97fcdb98632a91be79d3122b4eb99c0c4223ee")
                    .expect("couldn't parse data")
            );
            assert_eq!(
                receipt.gas_used,
                u64::from_str_radix("a0f9", 16).expect("couldn't parse data")
            );
            assert_eq!(receipt.logs.len(), 1);
            assert_eq!(receipt.state_root, None);
            assert_eq!(receipt.status, Some(true));
            assert_eq!(
                receipt.to,
                Some(
                    Address::from_str("dac17f958d2ee523a2206206994597c13d831ec7")
                        .expect("couldn't parse data")
                )
            );
            assert_eq!(receipt.transaction_hash, hash);
            assert_eq!(receipt.transaction_index, 136);
            assert_eq!(receipt.transaction_type, Some(0));
        }

        #[tokio::test]
        async fn get_storage_at_some() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let total_supply = TestRpcClient::new(&alchemy_url)
                .get_storage_at(
                    dai_address,
                    U256::from(1),
                    Some(BlockSpec::Number(16220843)),
                )
                .await
                .expect("should have succeeded");

            assert_eq!(
                total_supply,
                Some(
                    U256::from_str_radix(
                        "000000000000000000000000000000000000000010a596ae049e066d4991945c",
                        16
                    )
                    .expect("failed to parse storage location")
                )
            );
        }

        #[tokio::test]
        async fn get_storage_at_latest() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let _total_supply = TestRpcClient::new(&alchemy_url)
                .get_storage_at(
                    dai_address,
                    U256::from_str_radix(
                        "0000000000000000000000000000000000000000000000000000000000000001",
                        16,
                    )
                    .expect("failed to parse storage location"),
                    Some(BlockSpec::latest()),
                )
                .await
                .expect("should have succeeded");
        }

        #[tokio::test]
        async fn get_storage_at_future_block() {
            let alchemy_url = get_alchemy_url();

            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let storage_slot = TestRpcClient::new(&alchemy_url)
                .get_storage_at(
                    dai_address,
                    U256::from(1),
                    Some(BlockSpec::Number(MAX_BLOCK_NUMBER)),
                )
                .await
                .expect("should have succeeded");

            assert!(storage_slot.is_none());
        }

        #[tokio::test]
        async fn network_id_success() {
            let alchemy_url = get_alchemy_url();

            let version = TestRpcClient::new(&alchemy_url)
                .network_id()
                .await
                .expect("should have succeeded");

            assert_eq!(version, 1);
        }

        #[tokio::test]
        async fn stores_result_in_cache() {
            let alchemy_url = get_alchemy_url();
            let client = TestRpcClient::new(&alchemy_url);
            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            let total_supply = client
                .get_storage_at(
                    dai_address,
                    U256::from(1),
                    Some(BlockSpec::Number(16220843)),
                )
                .await
                .expect("should have succeeded");

            let cached_files = client.files_in_cache();
            assert_eq!(cached_files.len(), 1);

            let mut file = File::open(&cached_files[0]).expect("failed to open file");
            let cached_result: Option<U256> =
                serde_json::from_reader(&mut file).expect("failed to parse");

            assert_eq!(total_supply, cached_result);
        }

        #[tokio::test]
        async fn handles_invalid_type_in_cache_single_call() {
            let alchemy_url = get_alchemy_url();
            let client = TestRpcClient::new(&alchemy_url);
            let dai_address = Address::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .expect("failed to parse address");

            client
                .get_storage_at(
                    dai_address,
                    U256::from(1),
                    Some(BlockSpec::Number(16220843)),
                )
                .await
                .expect("should have succeeded");

            // Write some valid JSON, but invalid U256
            tokio::fs::write(&client.files_in_cache()[0], "\"not-hex\"")
                .await
                .unwrap();

            client
                .get_storage_at(
                    dai_address,
                    U256::from(1),
                    Some(BlockSpec::Number(16220843)),
                )
                .await
                .expect("should have succeeded");
        }
    }
}
