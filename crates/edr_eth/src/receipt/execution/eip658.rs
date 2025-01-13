use super::Eip658;
use crate::receipt::MapReceiptLogs;

impl<LogT, NewLogT> MapReceiptLogs<LogT, NewLogT, Eip658<NewLogT>> for Eip658<LogT> {
    fn map_logs(self, map_fn: impl FnMut(LogT) -> NewLogT) -> Eip658<NewLogT> {
        Eip658 {
            status: self.status,
            cumulative_gas_used: self.cumulative_gas_used,
            logs_bloom: self.logs_bloom,
            logs: self.logs.into_iter().map(map_fn).collect(),
        }
    }
}
