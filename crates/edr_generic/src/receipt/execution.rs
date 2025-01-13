use edr_eth::{
    l1,
    log::ExecutionLog,
    receipt::{
        execution::{Eip658, Legacy},
        Execution,
    },
    result::{ExecutionResult, HaltReason},
    transaction::TransactionType,
};
use edr_evm::{receipt::ExecutionReceiptBuilder, state::State};

use crate::{eip2718::TypedEnvelope, transaction};

pub struct Builder;

impl ExecutionReceiptBuilder<HaltReason, l1::SpecId, transaction::SignedWithFallbackToPostEip155>
    for Builder
{
    type Receipt = TypedEnvelope<Execution<ExecutionLog>>;

    fn new_receipt_builder<StateT: State>(
        _pre_execution_state: StateT,
        _transaction: &transaction::SignedWithFallbackToPostEip155,
    ) -> Result<Self, StateT::Error> {
        Ok(Self)
    }

    fn build_receipt(
        self,
        header: &edr_eth::block::PartialHeader,
        transaction: &crate::transaction::SignedWithFallbackToPostEip155,
        result: &ExecutionResult<HaltReason>,
        hardfork: l1::SpecId,
    ) -> Self::Receipt {
        let logs = result.logs().to_vec();
        let logs_bloom = edr_eth::log::logs_to_bloom(&logs);

        let receipt = if hardfork >= l1::SpecId::BYZANTIUM {
            Execution::Eip658(Eip658 {
                status: result.is_success(),
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        } else {
            Execution::Legacy(Legacy {
                root: header.state_root,
                cumulative_gas_used: header.gas_used,
                logs_bloom,
                logs,
            })
        };

        TypedEnvelope::new(receipt, transaction.transaction_type())
    }
}
