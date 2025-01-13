use core::fmt::Debug;
use std::{marker::PhantomData, sync::Arc};

use alloy_rlp::Encodable as _;
use derive_where::derive_where;
use edr_eth::{
    block::{self, Header, PartialHeader},
    keccak256, l1,
    log::{ExecutionLog, FilterLog, FullBlockLog, ReceiptLog},
    receipt::{MapReceiptLogs, ReceiptTrait, TransactionReceipt},
    spec::{ChainSpec, HardforkTrait},
    transaction::ExecutableTransaction,
    trie,
    withdrawal::Withdrawal,
    B256, KECCAK_EMPTY,
};
use edr_utils::types::TypeConstructor;
use itertools::izip;

use crate::{
    block::{BlockReceipts, EmptyBlock, LocalBlock},
    blockchain::BlockchainError,
    receipt::ReceiptFactory,
    spec::{
        ExecutionReceiptTypeConstructorBounds, ExecutionReceiptTypeConstructorForChainSpec,
        RuntimeSpec,
    },
    transaction::DetailedTransaction,
    Block,
};

/// Helper type for a local Ethereum block for a given chain spec.
pub type EthLocalBlockForChainSpec<ChainSpecT> = EthLocalBlock<
    <ChainSpecT as RuntimeSpec>::RpcBlockConversionError,
    <ChainSpecT as RuntimeSpec>::BlockReceipt,
    ExecutionReceiptTypeConstructorForChainSpec<ChainSpecT>,
    <ChainSpecT as ChainSpec>::Hardfork,
    <ChainSpecT as RuntimeSpec>::RpcReceiptConversionError,
    <ChainSpecT as ChainSpec>::SignedTransaction,
>;

/// A locally mined block, which contains complete information.
#[derive_where(Clone; SignedTransactionT)]
#[derive_where(Debug, PartialEq, Eq; BlockReceiptT, SignedTransactionT)]
pub struct EthLocalBlock<
    BlockConversionErrorT,
    BlockReceiptT: ReceiptTrait,
    ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
    HardforkT: HardforkTrait,
    ReceiptConversionErrorT,
    SignedTransactionT,
> {
    header: block::Header,
    transactions: Vec<SignedTransactionT>,
    transaction_receipts: Vec<Arc<BlockReceiptT>>,
    ommers: Vec<block::Header>,
    ommer_hashes: Vec<B256>,
    withdrawals: Option<Vec<Withdrawal>>,
    hash: B256,
    phantom: PhantomData<(
        BlockConversionErrorT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
    )>,
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: ReceiptTrait,
        HardforkT: HardforkTrait,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + ExecutableTransaction,
    >
    EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    /// Constructs a new instance with the provided data.
    pub fn new(
        receipt_factory: impl ReceiptFactory<
            <ExecutionReceiptTypeConstructorT as TypeConstructor<FilterLog>>::Type,
            HardforkT,
            SignedTransactionT,
            Output = BlockReceiptT,
        >,
        hardfork: HardforkT,
        partial_header: PartialHeader,
        transactions: Vec<SignedTransactionT>,
        transaction_receipts: Vec<
            TransactionReceipt<
                <ExecutionReceiptTypeConstructorT as TypeConstructor<ExecutionLog>>::Type,
            >,
        >,
        ommers: Vec<Header>,
        withdrawals: Option<Vec<Withdrawal>>,
    ) -> Self {
        let ommer_hashes = ommers.iter().map(Header::hash).collect::<Vec<_>>();
        let ommers_hash = keccak256(alloy_rlp::encode(&ommers));
        let transactions_root =
            trie::ordered_trie_root(transactions.iter().map(ExecutableTransaction::rlp_encoding));

        let withdrawals_root = withdrawals
            .as_ref()
            .map(|w| trie::ordered_trie_root(w.iter().map(alloy_rlp::encode)));

        let header = Header::new(
            partial_header,
            ommers_hash,
            transactions_root,
            withdrawals_root,
        );

        let hash = header.hash();
        let transaction_receipts =
            map_transaction_receipt_logs::<ExecutionReceiptTypeConstructorT>(
                hash,
                header.number,
                transaction_receipts,
            )
            .zip(transactions.iter())
            .map(|(transaction_receipt, transaction)| {
                Arc::new(receipt_factory.create_receipt(
                    hardfork,
                    transaction,
                    transaction_receipt,
                    &hash,
                    header.number,
                ))
            })
            .collect();

        Self {
            header,
            transactions,
            transaction_receipts,
            ommers,
            ommer_hashes,
            withdrawals,
            hash,
            phantom: PhantomData,
        }
    }

    /// Retrieves the block's transactions.
    pub fn detailed_transactions(
        &self,
    ) -> impl Iterator<Item = DetailedTransaction<'_, SignedTransactionT, Arc<BlockReceiptT>>> {
        izip!(self.transactions.iter(), self.transaction_receipts.iter()).map(
            |(transaction, receipt)| DetailedTransaction {
                transaction,
                receipt,
            },
        )
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: Debug + ReceiptTrait + alloy_rlp::Encodable,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + alloy_rlp::Encodable,
    >
    EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    fn rlp_payload_length(&self) -> usize {
        self.header.length()
            + self.transactions.length()
            + self.ommers.length()
            + self
                .withdrawals
                .as_ref()
                .map_or(0, alloy_rlp::Encodable::length)
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: Debug + ReceiptTrait + alloy_rlp::Encodable,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + alloy_rlp::Encodable,
    > Block<SignedTransactionT>
    for EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    fn block_hash(&self) -> &B256 {
        &self.hash
    }

    fn header(&self) -> &block::Header {
        &self.header
    }

    fn rlp_size(&self) -> u64 {
        alloy_rlp::encode(self)
            .len()
            .try_into()
            .expect("usize fits into u64")
    }

    fn transactions(&self) -> &[SignedTransactionT] {
        &self.transactions
    }

    fn ommer_hashes(&self) -> &[B256] {
        self.ommer_hashes.as_slice()
    }

    fn withdrawals(&self) -> Option<&[Withdrawal]> {
        self.withdrawals.as_deref()
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: ReceiptTrait + Debug + alloy_rlp::Encodable,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + alloy_rlp::Encodable,
    > BlockReceipts<Arc<BlockReceiptT>>
    for EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    type Error = BlockchainError<BlockConversionErrorT, HardforkT, ReceiptConversionErrorT>;

    fn fetch_transaction_receipts(
        &self,
    ) -> Result<
        Vec<Arc<BlockReceiptT>>,
        BlockchainError<BlockConversionErrorT, HardforkT, ReceiptConversionErrorT>,
    > {
        Ok(self.transaction_receipts.clone())
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: ReceiptTrait,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + ExecutableTransaction + alloy_rlp::Encodable,
    > EmptyBlock<HardforkT>
    for EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    fn empty(hardfork: HardforkT, partial_header: PartialHeader) -> Self {
        let (withdrawals, withdrawals_root) = if hardfork.into() >= l1::SpecId::SHANGHAI {
            Some((Vec::new(), KECCAK_EMPTY))
        } else {
            None
        }
        .unzip();

        let header = Header::new(partial_header, KECCAK_EMPTY, KECCAK_EMPTY, withdrawals_root);
        let hash = header.hash();

        Self {
            header,
            transactions: Vec::new(),
            transaction_receipts: Vec::new(),
            ommers: Vec::new(),
            ommer_hashes: Vec::new(),
            withdrawals,
            hash,
            phantom: PhantomData,
        }
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: ReceiptTrait,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + ExecutableTransaction + alloy_rlp::Encodable,
    > LocalBlock<Arc<BlockReceiptT>>
    for EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    fn transaction_receipts(&self) -> &[Arc<BlockReceiptT>] {
        &self.transaction_receipts
    }
}

impl<
        BlockConversionErrorT,
        BlockReceiptT: Debug + ReceiptTrait + alloy_rlp::Encodable,
        ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
        HardforkT: HardforkTrait,
        ReceiptConversionErrorT,
        SignedTransactionT: Debug + alloy_rlp::Encodable,
    > alloy_rlp::Encodable
    for EthLocalBlock<
        BlockConversionErrorT,
        BlockReceiptT,
        ExecutionReceiptTypeConstructorT,
        HardforkT,
        ReceiptConversionErrorT,
        SignedTransactionT,
    >
{
    fn encode(&self, out: &mut dyn alloy_rlp::BufMut) {
        alloy_rlp::Header {
            list: true,
            payload_length: self.rlp_payload_length(),
        }
        .encode(out);

        self.header.encode(out);
        self.transactions.encode(out);
        self.ommers.encode(out);

        if let Some(withdrawals) = &self.withdrawals {
            withdrawals.encode(out);
        }
    }

    fn length(&self) -> usize {
        let payload_length = self.rlp_payload_length();
        payload_length + alloy_rlp::length_of_length(payload_length)
    }
}

/// Maps the logs of the transaction receipts from [`ExecutionLog`] to
/// [`FilterLog`].
fn map_transaction_receipt_logs<
    ExecutionReceiptTypeConstructorT: ExecutionReceiptTypeConstructorBounds,
>(
    block_hash: B256,
    block_number: u64,
    receipts: Vec<
        TransactionReceipt<
            <ExecutionReceiptTypeConstructorT as TypeConstructor<ExecutionLog>>::Type,
        >,
    >,
) -> impl Iterator<
    Item = TransactionReceipt<
        <ExecutionReceiptTypeConstructorT as TypeConstructor<FilterLog>>::Type,
    >,
> {
    let mut log_index = 0;

    receipts
        .into_iter()
        .enumerate()
        .map(move |(transaction_index, receipt)| {
            let transaction_index = transaction_index as u64;
            let transaction_hash = receipt.transaction_hash;

            receipt.map_logs(|log| {
                FilterLog {
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
                    // Assuming a local block is never reorged out.
                    removed: false,
                }
            })
        })
}
