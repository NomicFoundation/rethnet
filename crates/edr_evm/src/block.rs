mod builder;
mod local;
mod remote;
/// Block-related transaction types.
pub mod transaction;

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use auto_impl::auto_impl;
use edr_eth::{
    block::{self, BlobGas, Header, PartialHeader},
    receipt::ReceiptTrait,
    spec::{ChainSpec, HardforkTrait},
    transaction::ExecutableTransaction,
    withdrawal::Withdrawal,
    B256, U256,
};

pub use self::{
    builder::{
        BlockBuilder, BlockBuilderAndError, BlockBuilderCreationError, BlockTransactionError,
        EthBlockBuilder, EthBlockReceiptFactory,
    },
    local::{EthLocalBlock, EthLocalBlockForChainSpec},
    remote::{ConversionError as RemoteBlockConversionError, EthRpcBlock, RemoteBlock},
};
use crate::spec::RuntimeSpec;

/// Trait for implementations of an Ethereum block.
#[auto_impl(Arc)]
pub trait Block<SignedTransactionT>: Debug {
    /// Returns the block's hash.
    fn block_hash(&self) -> &B256;

    /// Returns the block's header.
    fn header(&self) -> &block::Header;

    /// Ommer/uncle block hashes.
    fn ommer_hashes(&self) -> &[B256];

    /// The length of the RLP encoding of this block in bytes.
    fn rlp_size(&self) -> u64;

    /// Returns the block's transactions.
    fn transactions(&self) -> &[SignedTransactionT];

    /// Withdrawals
    fn withdrawals(&self) -> Option<&[Withdrawal]>;
}

/// Trait for fetching the receipts of a block's transactions.
#[auto_impl(Arc)]
pub trait BlockReceipts<BlockReceiptT: ReceiptTrait> {
    /// The blockchain error type.
    type Error;

    /// Fetches the receipts of the block's transactions.
    ///
    /// This may block if the receipts are stored remotely.
    fn fetch_transaction_receipts(&self) -> Result<Vec<BlockReceiptT>, Self::Error>;
}

/// Trait for creating an empty block.
pub trait EmptyBlock<HardforkT: HardforkTrait> {
    /// Constructs an empty block.
    fn empty(hardfork: HardforkT, partial_header: PartialHeader) -> Self;
}

impl<BlockT: EmptyBlock<HardforkT>, HardforkT: HardforkTrait> EmptyBlock<HardforkT>
    for Arc<BlockT>
{
    fn empty(hardfork: HardforkT, partial_header: PartialHeader) -> Self {
        Arc::new(BlockT::empty(hardfork, partial_header))
    }
}

/// Trait for locally mined blocks.
#[auto_impl(Arc)]
pub trait LocalBlock<BlockReceiptT> {
    /// Returns the receipts of the block's transactions.
    fn transaction_receipts(&self) -> &[BlockReceiptT];
}

/// Trait that meets all requirements for an Ethereum block.
pub trait EthBlock<BlockReceiptT: ReceiptTrait, SignedTransactionT>:
    Block<SignedTransactionT> + BlockReceipts<BlockReceiptT>
{
}

impl<BlockReceiptT, BlockT, SignedTransactionT> EthBlock<BlockReceiptT, SignedTransactionT>
    for BlockT
where
    BlockReceiptT: ReceiptTrait,
    BlockT: Block<SignedTransactionT> + BlockReceipts<BlockReceiptT>,
{
}

/// Trait that meets all requirements for a synchronous block.
pub trait SyncBlock<BlockReceiptT: ReceiptTrait, SignedTransactionT>:
    EthBlock<BlockReceiptT, SignedTransactionT> + Send + Sync
{
}

impl<BlockReceiptT, BlockT, SignedTransactionT> SyncBlock<BlockReceiptT, SignedTransactionT>
    for BlockT
where
    BlockReceiptT: ReceiptTrait,
    BlockT: EthBlock<BlockReceiptT, SignedTransactionT> + Send + Sync,
{
}

/// A type containing the relevant data for an Ethereum block.
pub struct EthBlockData<ChainSpecT: RuntimeSpec> {
    /// The block's header.
    pub header: edr_eth::block::Header,
    /// The block's transactions.
    pub transactions: Vec<ChainSpecT::SignedTransaction>,
    /// The hashes of the block's ommers.
    pub ommer_hashes: Vec<B256>,
    /// The staking withdrawals.
    pub withdrawals: Option<Vec<Withdrawal>>,
    /// The block's hash.
    pub hash: B256,
    /// The length of the RLP encoding of this block in bytes.
    pub rlp_size: u64,
}

impl<ChainSpecT: RuntimeSpec> TryFrom<edr_rpc_eth::Block<ChainSpecT::RpcTransaction>>
    for EthBlockData<ChainSpecT>
{
    type Error = RemoteBlockConversionError<ChainSpecT::RpcTransactionConversionError>;

    fn try_from(
        value: edr_rpc_eth::Block<ChainSpecT::RpcTransaction>,
    ) -> Result<Self, Self::Error> {
        let header = Header {
            parent_hash: value.parent_hash,
            ommers_hash: value.sha3_uncles,
            beneficiary: value
                .miner
                .ok_or(RemoteBlockConversionError::MissingMiner)?,
            state_root: value.state_root,
            transactions_root: value.transactions_root,
            receipts_root: value.receipts_root,
            logs_bloom: value.logs_bloom,
            difficulty: value.difficulty,
            number: value
                .number
                .ok_or(RemoteBlockConversionError::MissingNumber)?,
            gas_limit: value.gas_limit,
            gas_used: value.gas_used,
            timestamp: value.timestamp,
            extra_data: value.extra_data,
            // TODO don't accept remote blocks with missing mix hash,
            // see https://github.com/NomicFoundation/edr/issues/518
            mix_hash: value.mix_hash.unwrap_or_default(),
            nonce: value
                .nonce
                .ok_or(RemoteBlockConversionError::MissingNonce)?,
            base_fee_per_gas: value.base_fee_per_gas,
            withdrawals_root: value.withdrawals_root,
            blob_gas: value.blob_gas_used.and_then(|gas_used| {
                value.excess_blob_gas.map(|excess_gas| BlobGas {
                    gas_used,
                    excess_gas,
                })
            }),
            parent_beacon_block_root: value.parent_beacon_block_root,
        };

        let transactions = value
            .transactions
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<ChainSpecT::SignedTransaction>, _>>()
            .map_err(RemoteBlockConversionError::TransactionConversionError)?;

        let hash = value.hash.ok_or(RemoteBlockConversionError::MissingHash)?;

        Ok(Self {
            header,
            transactions,
            ommer_hashes: value.uncles,
            withdrawals: value.withdrawals,
            hash,
            rlp_size: value.size,
        })
    }
}

/// Helper type for a chain-specific [`BlockAndTotalDifficulty`].
pub type BlockAndTotalDifficultyForChainSpec<ChainSpecT> = BlockAndTotalDifficulty<
    Arc<<ChainSpecT as RuntimeSpec>::Block>,
    <ChainSpecT as ChainSpec>::SignedTransaction,
>;

/// The result returned by requesting a block by number.
#[derive(Clone, Debug)]
pub struct BlockAndTotalDifficulty<BlockT, SignedTransactionT> {
    /// The block
    pub block: BlockT,
    /// The total difficulty with the block
    pub total_difficulty: Option<U256>,
    phantom: PhantomData<SignedTransactionT>,
}

impl<BlockT, SignedTransactionT> BlockAndTotalDifficulty<BlockT, SignedTransactionT> {
    /// Creates a new block and total difficulty.
    pub fn new(block: BlockT, total_difficulty: Option<U256>) -> Self {
        Self {
            block,
            total_difficulty,
            phantom: PhantomData,
        }
    }
}

impl<BlockT: Block<SignedTransactionT>, SignedTransactionT>
    From<BlockAndTotalDifficulty<BlockT, SignedTransactionT>> for edr_rpc_eth::Block<B256>
where
    SignedTransactionT: ExecutableTransaction,
{
    fn from(value: BlockAndTotalDifficulty<BlockT, SignedTransactionT>) -> Self {
        let transactions = value
            .block
            .transactions()
            .iter()
            .map(|tx| *tx.transaction_hash())
            .collect();

        let header = value.block.header();
        edr_rpc_eth::Block {
            hash: Some(*value.block.block_hash()),
            parent_hash: header.parent_hash,
            sha3_uncles: header.ommers_hash,
            state_root: header.state_root,
            transactions_root: header.transactions_root,
            receipts_root: header.receipts_root,
            number: Some(header.number),
            gas_used: header.gas_used,
            gas_limit: header.gas_limit,
            extra_data: header.extra_data.clone(),
            logs_bloom: header.logs_bloom,
            timestamp: header.timestamp,
            difficulty: header.difficulty,
            total_difficulty: value.total_difficulty,
            uncles: value.block.ommer_hashes().to_vec(),
            transactions,
            size: value.block.rlp_size(),
            mix_hash: Some(header.mix_hash),
            nonce: Some(header.nonce),
            base_fee_per_gas: header.base_fee_per_gas,
            miner: Some(header.beneficiary),
            withdrawals: value
                .block
                .withdrawals()
                .map(<[edr_eth::withdrawal::Withdrawal]>::to_vec),
            withdrawals_root: header.withdrawals_root,
            blob_gas_used: header.blob_gas.as_ref().map(|bg| bg.gas_used),
            excess_blob_gas: header.blob_gas.as_ref().map(|bg| bg.excess_gas),
            parent_beacon_block_root: header.parent_beacon_block_root,
        }
    }
}
