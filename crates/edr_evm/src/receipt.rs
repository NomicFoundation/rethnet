// Re-export the receipt types from `edr_eth`.
pub use edr_eth::receipt::*;
use edr_eth::{
    block,
    eips::eip2718::TypedEnvelope,
    l1,
    log::{self, ExecutionLog},
    receipt,
    result::ExecutionResult,
    spec::{HaltReasonTrait, HardforkTrait},
    transaction::{self, Transaction, TransactionType as _, TransactionValidation},
};

use crate::state::State;

/// Trait for a builder that constructs an execution receipt.
pub trait ExecutionReceiptBuilder<HaltReasonT, HardforkT, TransactionT>: Sized
where
    HaltReasonT: HaltReasonTrait,
    HardforkT: HardforkTrait,
    TransactionT: Transaction + TransactionValidation,
{
    /// The receipt type that the builder constructs.
    type Receipt;

    /// Creates a new builder with the given pre-execution state.
    fn new_receipt_builder<StateT: State>(
        pre_execution_state: StateT,
        transaction: &TransactionT,
    ) -> Result<Self, StateT::Error>;

    /// Builds a receipt using the provided information.
    fn build_receipt(
        self,
        header: &block::PartialHeader,
        transaction: &TransactionT,
        result: &ExecutionResult<HaltReasonT>,
        hardfork: HardforkT,
    ) -> Self::Receipt;
}

/// Builder for execution receipts.
pub struct Builder;

impl ExecutionReceiptBuilder<l1::HaltReason, l1::SpecId, transaction::Signed> for Builder {
    type Receipt = TypedEnvelope<receipt::Execution<ExecutionLog>>;

    fn new_receipt_builder<StateT: State>(
        _pre_execution_state: StateT,
        _transaction: &transaction::Signed,
    ) -> Result<Self, StateT::Error> {
        Ok(Self)
    }

    fn build_receipt(
        self,
        header: &block::PartialHeader,
        transaction: &transaction::Signed,
        result: &ExecutionResult<l1::HaltReason>,
        hardfork: l1::SpecId,
    ) -> Self::Receipt {
        let logs = result.logs().to_vec();
        let logs_bloom = log::logs_to_bloom(&logs);

        let receipt = if hardfork >= l1::SpecId::BYZANTIUM {
            receipt::Execution::Eip658(receipt::execution::Eip658 {
                status: result.is_success(),
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        } else {
            receipt::Execution::Legacy(receipt::execution::Legacy {
                root: header.state_root,
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        };

        TypedEnvelope::new(receipt, transaction.transaction_type())
    }
}
