use edr_eth::{
    log::FilterLog,
    receipt::{ReceiptFactory, TransactionReceipt},
    B256,
};
use edr_evm::EthBlockReceiptFactory;
use revm_optimism::{L1BlockInfo, OptimismTransaction as _};

use crate::{eip2718::TypedEnvelope, receipt, transaction, OptimismSpecId};

/// Block receipt factory for Optimism.
pub struct BlockReceiptFactory {
    pub(crate) l1_block_info: L1BlockInfo,
}

impl
    ReceiptFactory<
        TypedEnvelope<receipt::Execution<FilterLog>>,
        OptimismSpecId,
        transaction::Signed,
    > for BlockReceiptFactory
{
    type Output = receipt::Block;

    fn create_receipt(
        &self,
        hardfork: OptimismSpecId,
        transaction: &transaction::Signed,
        transaction_receipt: TransactionReceipt<TypedEnvelope<receipt::Execution<FilterLog>>>,
        block_hash: &B256,
        block_number: u64,
    ) -> Self::Output {
        let l1_block_info = to_rpc_l1_block_info(
            hardfork,
            &self.l1_block_info,
            transaction,
            &transaction_receipt,
        );

        let eth = {
            let receipt_factory = EthBlockReceiptFactory::default();
            receipt_factory.create_receipt(
                hardfork,
                transaction,
                transaction_receipt,
                block_hash,
                block_number,
            )
        };

        receipt::Block { eth, l1_block_info }
    }
}

fn to_rpc_l1_block_info(
    hardfork: OptimismSpecId,
    l1_block_info: &L1BlockInfo,
    transaction: &transaction::Signed,
    transaction_receipt: &TransactionReceipt<TypedEnvelope<receipt::Execution<FilterLog>>>,
) -> Option<op_alloy_rpc_types::receipt::L1BlockInfo> {
    if matches!(
        transaction_receipt.inner,
        TypedEnvelope::Legacy(_) | TypedEnvelope::Deposit(_)
    ) {
        None
    } else {
        let enveloped_tx = transaction
            .enveloped_tx()
            .expect("Non-deposit transactions must return an enveloped transaction");

        let l1_fee = l1_block_info.calculate_tx_l1_cost(&enveloped_tx, hardfork);

        let (l1_fee_scalar, l1_base_fee_scalar) = if hardfork < OptimismSpecId::ECOTONE {
            let l1_fee_scalar: f64 = l1_block_info.l1_base_fee_scalar.into();

            (Some(l1_fee_scalar / 1_000_000f64), None)
        } else {
            let l1_base_fee_scalar = l1_block_info
                .l1_base_fee_scalar
                .try_into()
                .expect("L1 base fee scalar cannot be larger than u128::max");

            (None, Some(l1_base_fee_scalar))
        };

        let l1_gas_used = l1_block_info
            .data_gas(&enveloped_tx, hardfork)
            .saturating_add(l1_block_info.l1_fee_overhead.unwrap_or_default());

        let l1_block_info = op_alloy_rpc_types::receipt::L1BlockInfo {
            l1_gas_price: Some(
                l1_block_info
                    .l1_base_fee
                    .try_into()
                    .expect("L1 gas price cannot be larger than u128::max"),
            ),
            l1_gas_used: Some(
                l1_gas_used
                    .try_into()
                    .expect("L1 gas used cannot be larger than u128::max"),
            ),
            l1_fee: Some(
                l1_fee
                    .try_into()
                    .expect("L1 fee cannot be larger than u128::max"),
            ),
            l1_fee_scalar,
            l1_base_fee_scalar,
            l1_blob_base_fee: l1_block_info.l1_blob_base_fee.map(|scalar| {
                scalar
                    .try_into()
                    .expect("L1 blob base fee cannot be larger than u128::max")
            }),
            l1_blob_base_fee_scalar: l1_block_info.l1_blob_base_fee_scalar.map(|scalar| {
                scalar
                    .try_into()
                    .expect("L1 blob base fee scalar cannot be larger than u128::max")
            }),
        };

        Some(l1_block_info)
    }
}
