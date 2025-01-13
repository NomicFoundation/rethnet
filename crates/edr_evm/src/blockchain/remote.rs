use std::sync::Arc;

use async_rwlock::{RwLock, RwLockUpgradableReadGuard};
use derive_where::derive_where;
use edr_eth::{
    filter::OneOrMore, log::FilterLog, Address, BlockSpec, PreEip1898BlockSpec, B256, U256,
};
use edr_rpc_eth::client::EthRpcClient;
use revm::primitives::HashSet;
use tokio::runtime;

use super::{forked::ForkedBlockchainErrorForChainSpec, storage::SparseBlockchainStorage};
use crate::{
    blockchain::ForkedBlockchainError, spec::RuntimeSpec,
    transaction::remote::EthRpcTransaction as _, Block, EthRpcBlock as _, RemoteBlock,
};

#[derive_where(Debug; BlockT)]
pub struct RemoteBlockchain<BlockT, ChainSpecT, const FORCE_CACHING: bool>
where
    BlockT: Block<ChainSpecT::SignedTransaction> + Clone,
    ChainSpecT: RuntimeSpec,
{
    client: Arc<EthRpcClient<ChainSpecT>>,
    cache: RwLock<
        SparseBlockchainStorage<
            Arc<ChainSpecT::BlockReceipt>,
            BlockT,
            ChainSpecT::SignedTransaction,
        >,
    >,
    runtime: runtime::Handle,
}

impl<BlockT, ChainSpecT, const FORCE_CACHING: bool>
    RemoteBlockchain<BlockT, ChainSpecT, FORCE_CACHING>
where
    BlockT: Block<ChainSpecT::SignedTransaction> + Clone,
    ChainSpecT: RuntimeSpec,
{
    /// Constructs a new instance with the provided RPC client.
    pub fn new(client: Arc<EthRpcClient<ChainSpecT>>, runtime: runtime::Handle) -> Self {
        Self {
            client,
            cache: RwLock::new(SparseBlockchainStorage::default()),
            runtime,
        }
    }

    /// Retrieves the instance's RPC client.
    pub fn client(&self) -> &Arc<EthRpcClient<ChainSpecT>> {
        &self.client
    }

    pub async fn logs(
        &self,
        from_block: BlockSpec,
        to_block: BlockSpec,
        addresses: &HashSet<Address>,
        normalized_topics: &[Option<Vec<B256>>],
    ) -> Result<Vec<FilterLog>, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        self.client
            .get_logs_by_range(
                from_block,
                to_block,
                if addresses.len() > 1 {
                    Some(OneOrMore::Many(addresses.iter().copied().collect()))
                } else {
                    addresses
                        .iter()
                        .next()
                        .map(|address| OneOrMore::One(*address))
                },
                if normalized_topics.is_empty() {
                    None
                } else {
                    Some(
                        normalized_topics
                            .iter()
                            .map(|topics| {
                                topics.as_ref().and_then(|topics| {
                                    if topics.len() > 1 {
                                        Some(OneOrMore::Many(topics.clone()))
                                    } else {
                                        topics.first().map(|topic| OneOrMore::One(*topic))
                                    }
                                })
                            })
                            .collect(),
                    )
                },
            )
            .await
            .map_err(ForkedBlockchainError::RpcClient)
    }

    /// Retrieves the receipt of the transaction with the provided hash, if it
    /// exists.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub async fn receipt_by_transaction_hash(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<Arc<ChainSpecT::BlockReceipt>>, ForkedBlockchainErrorForChainSpec<ChainSpecT>>
    {
        let cache = self.cache.upgradable_read().await;

        if let Some(receipt) = cache.receipt_by_transaction_hash(transaction_hash) {
            Ok(Some(receipt.clone()))
        } else if let Some(receipt) = self
            .client
            .get_transaction_receipt(*transaction_hash)
            .await?
        {
            let receipt = receipt
                .try_into()
                .map_err(ForkedBlockchainError::ReceiptConversion)?;

            Ok(Some({
                let mut cache = RwLockUpgradableReadGuard::upgrade(cache).await;
                cache.insert_receipt(Arc::new(receipt))?.clone()
            }))
        } else {
            Ok(None)
        }
    }

    /// Retrieves the blockchain's runtime.
    pub fn runtime(&self) -> &runtime::Handle {
        &self.runtime
    }
}

impl<BlockT, ChainSpecT, const FORCE_CACHING: bool>
    RemoteBlockchain<BlockT, ChainSpecT, FORCE_CACHING>
where
    BlockT: Block<ChainSpecT::SignedTransaction> + Clone + From<RemoteBlock<ChainSpecT>>,
    ChainSpecT: RuntimeSpec,
{
    /// Retrieves the block with the provided hash, if it exists.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub async fn block_by_hash(
        &self,
        hash: &B256,
    ) -> Result<Option<BlockT>, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        let cache = self.cache.upgradable_read().await;

        if let Some(block) = cache.block_by_hash(hash).cloned() {
            return Ok(Some(block));
        }

        if let Some(block) = self
            .client
            .get_block_by_hash_with_transaction_data(*hash)
            .await?
        {
            self.fetch_and_cache_block(cache, block)
                .await
                .map(Option::Some)
        } else {
            Ok(None)
        }
    }

    /// Retrieves the block with the provided number, if it exists.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub async fn block_by_number(
        &self,
        number: u64,
    ) -> Result<BlockT, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        let cache = self.cache.upgradable_read().await;

        if let Some(block) = cache.block_by_number(number).cloned() {
            Ok(block)
        } else {
            let block = self
                .client
                .get_block_by_number_with_transaction_data(PreEip1898BlockSpec::Number(number))
                .await?;

            self.fetch_and_cache_block(cache, block).await
        }
    }

    /// Retrieves the block that contains a transaction with the provided hash,
    /// if it exists.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub async fn block_by_transaction_hash(
        &self,
        transaction_hash: &B256,
    ) -> Result<Option<BlockT>, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        // This block ensure that the read lock is dropped
        {
            if let Some(block) = self
                .cache
                .read()
                .await
                .block_by_transaction_hash(transaction_hash)
                .cloned()
            {
                return Ok(Some(block));
            }
        }

        if let Some(transaction) = self
            .client
            .get_transaction_by_hash(*transaction_hash)
            .await?
        {
            self.block_by_hash(transaction.block_hash().expect("Not a pending transaction"))
                .await
        } else {
            Ok(None)
        }
    }

    /// Retrieves the total difficulty at the block with the provided hash.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    pub async fn total_difficulty_by_hash(
        &self,
        hash: &B256,
    ) -> Result<Option<U256>, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        let cache = self.cache.upgradable_read().await;

        if let Some(difficulty) = cache.total_difficulty_by_hash(hash).cloned() {
            Ok(Some(difficulty))
        } else if let Some(block) = self
            .client
            .get_block_by_hash_with_transaction_data(*hash)
            .await?
        {
            // Geth has recently removed the total difficulty field from block RPC
            // responses, so we fall back to the terminal total difficulty of main net to
            // provide backwards compatibility.
            // TODO https://github.com/NomicFoundation/edr/issues/696
            let total_difficulty = *block
                .total_difficulty()
                .unwrap_or(&edr_defaults::TERMINAL_TOTAL_DIFFICULTY);

            self.fetch_and_cache_block(cache, block).await?;

            Ok(Some(total_difficulty))
        } else {
            Ok(None)
        }
    }

    /// Fetches detailed block information and caches the block.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    async fn fetch_and_cache_block(
        &self,
        cache: RwLockUpgradableReadGuard<
            '_,
            SparseBlockchainStorage<
                Arc<ChainSpecT::BlockReceipt>,
                BlockT,
                ChainSpecT::SignedTransaction,
            >,
        >,
        block: ChainSpecT::RpcBlock<ChainSpecT::RpcTransaction>,
    ) -> Result<BlockT, ForkedBlockchainErrorForChainSpec<ChainSpecT>> {
        // Geth has recently removed the total difficulty field from block RPC
        // responses, so we fall back to the terminal total difficulty of main net to
        // provide backwards compatibility.
        // TODO https://github.com/NomicFoundation/edr/issues/696
        let total_difficulty = *block
            .total_difficulty()
            .unwrap_or(&edr_defaults::TERMINAL_TOTAL_DIFFICULTY);

        let block = RemoteBlock::new(block, self.client.clone(), self.runtime.clone())
            .map_err(ForkedBlockchainError::BlockCreation)?;

        let is_cacheable = FORCE_CACHING
            || self
                .client
                .is_cacheable_block_number(block.header().number)
                .await?;

        let block = BlockT::from(block);

        if is_cacheable {
            let mut remote_cache = RwLockUpgradableReadGuard::upgrade(cache).await;

            Ok(remote_cache.insert_block(block, total_difficulty)?.clone())
        } else {
            Ok(block)
        }
    }
}

#[cfg(all(test, feature = "test-remote"))]
mod tests {
    use edr_eth::l1::L1ChainSpec;
    use edr_test_utils::env::get_alchemy_url;

    use super::*;

    #[tokio::test]
    async fn no_cache_for_unsafe_block_number() {
        let tempdir = tempfile::tempdir().expect("can create tempdir");

        let rpc_client = EthRpcClient::<L1ChainSpec>::new(
            &get_alchemy_url(),
            tempdir.path().to_path_buf(),
            None,
        )
        .expect("url ok");

        // Latest block number is always unsafe to cache
        let block_number = rpc_client.block_number().await.unwrap();

        let remote = RemoteBlockchain::<RemoteBlock<L1ChainSpec>, L1ChainSpec, false>::new(
            Arc::new(rpc_client),
            runtime::Handle::current(),
        );

        let _ = remote.block_by_number(block_number).await.unwrap();
        assert!(remote
            .cache
            .read()
            .await
            .block_by_number(block_number)
            .is_none());
    }
}
