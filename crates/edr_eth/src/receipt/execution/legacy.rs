use super::Legacy;
use crate::receipt::MapReceiptLogs;

impl<LogT, NewLogT> MapReceiptLogs<LogT, NewLogT, Legacy<NewLogT>> for Legacy<LogT> {
    fn map_logs(self, map_fn: impl FnMut(LogT) -> NewLogT) -> Legacy<NewLogT> {
        Legacy {
            root: self.root,
            cumulative_gas_used: self.cumulative_gas_used,
            logs_bloom: self.logs_bloom,
            logs: self.logs.into_iter().map(map_fn).collect(),
        }
    }
}
