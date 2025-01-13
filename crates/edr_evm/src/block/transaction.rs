use std::sync::Arc;

use edr_eth::{
    l1::{self, L1ChainSpec},
    spec::ChainSpec,
    transaction::SignedTransaction as _,
};
use edr_rpc_eth::RpcTypeFrom;

use crate::spec::RuntimeSpec;

/// Helper type for a chain-specific [`TransactionAndBlock`].
pub type TransactionAndBlockForChainSpec<ChainSpecT> = TransactionAndBlock<
    Arc<<ChainSpecT as RuntimeSpec>::Block>,
    <ChainSpecT as ChainSpec>::SignedTransaction,
>;

/// The result returned by requesting a transaction.
#[derive(Clone, Debug)]
pub struct TransactionAndBlock<BlockT, SignedTransactionT> {
    /// The transaction.
    pub transaction: SignedTransactionT,
    /// Block data in which the transaction is found if it has been mined.
    pub block_data: Option<BlockDataForTransaction<BlockT>>,
    /// Whether the transaction is pending
    pub is_pending: bool,
}

/// Block metadata for a transaction.
#[derive(Clone, Debug)]
pub struct BlockDataForTransaction<BlockT> {
    /// The block in which the transaction is found.
    pub block: BlockT,
    /// The index of the transaction in the block.
    pub transaction_index: u64,
}

impl RpcTypeFrom<TransactionAndBlockForChainSpec<L1ChainSpec>>
    for edr_rpc_eth::TransactionWithSignature
{
    type Hardfork = l1::SpecId;

    fn rpc_type_from(
        value: &TransactionAndBlockForChainSpec<L1ChainSpec>,
        hardfork: Self::Hardfork,
    ) -> Self {
        let (header, transaction_index) = value
            .block_data
            .as_ref()
            .map(
                |BlockDataForTransaction {
                     block,
                     transaction_index,
                 }| (block.header(), *transaction_index),
            )
            .unzip();

        let transaction = edr_rpc_eth::Transaction::new(
            &value.transaction,
            header,
            transaction_index,
            value.is_pending,
            hardfork,
        );
        let signature = value.transaction.signature();

        edr_rpc_eth::TransactionWithSignature::new(
            transaction,
            signature.r(),
            signature.s(),
            signature.v(),
            signature.y_parity(),
        )
    }
}
