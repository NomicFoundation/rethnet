use core::fmt::Debug;
use std::{fmt::Display, marker::PhantomData, sync::Arc};

use ansi_term::{Color, Style};
use derive_where::derive_where;
use edr_eth::{result::ExecutionResult, transaction::ExecutableTransaction, Bytes, B256, U256};
use edr_evm::{
    blockchain::BlockchainErrorForChainSpec,
    precompile::{self, Precompiles},
    trace::{AfterMessage, Trace, TraceMessage},
    transaction::Transaction as _,
    Block as _,
};
use edr_provider::{
    time::CurrentTime, CallResult, DebugMineBlockResult, DebugMineBlockResultForChainSpec,
    EstimateGasFailure, ProviderError, ProviderSpec, TransactionFailure,
};
use edr_solidity::contract_decoder::{ContractAndFunctionName, ContractDecoder};
use itertools::izip;

/// Trait for a function that decodes console log inputs.
pub trait DecodeConsoleLogInputsFn: Fn(Vec<Bytes>) -> Vec<String> + Send + Sync {}

impl<FnT> DecodeConsoleLogInputsFn for FnT where FnT: Fn(Vec<Bytes>) -> Vec<String> + Send + Sync {}

/// Trait for a function that prints a line or replaces the last printed line.
pub trait PrintLineFn: Fn(String, bool) -> Result<(), LoggerError> + Send + Sync {}

impl<FnT> PrintLineFn for FnT where FnT: Fn(String, bool) -> Result<(), LoggerError> + Send + Sync {}

#[derive(Clone)]
pub struct Config {
    /// Whether to enable the logger.
    pub enable: bool,
    pub decode_console_log_inputs_fn: Arc<dyn DecodeConsoleLogInputsFn>,
    pub print_line_fn: Arc<dyn PrintLineFn>,
}

#[derive(Clone)]
pub enum LoggingState {
    CollapsingMethod(CollapsedMethod),
    HardhatMinining {
        empty_blocks_range_start: Option<u64>,
    },
    IntervalMining {
        empty_blocks_range_start: Option<u64>,
    },
    Empty,
}

impl LoggingState {
    /// Converts the state into a hardhat mining state.
    pub fn into_hardhat_mining(self) -> Option<u64> {
        match self {
            Self::HardhatMinining {
                empty_blocks_range_start,
            } => empty_blocks_range_start,
            _ => None,
        }
    }

    /// Converts the state into an interval mining state.
    pub fn into_interval_mining(self) -> Option<u64> {
        match self {
            Self::IntervalMining {
                empty_blocks_range_start,
            } => empty_blocks_range_start,
            _ => None,
        }
    }
}

impl Default for LoggingState {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Clone)]
enum LogLine {
    Single(String),
    WithTitle(String, String),
}

#[derive(Debug, thiserror::Error)]
pub enum LoggerError {
    #[error("Failed to print line")]
    PrintLine,
}

#[derive_where(Clone)]
pub struct Logger<ChainSpecT: ProviderSpec<CurrentTime>> {
    collector: LogCollector<ChainSpecT>,
}

impl<ChainSpecT: ProviderSpec<CurrentTime>> Logger<ChainSpecT> {
    pub fn new(config: Config, contract_decoder: Arc<ContractDecoder>) -> napi::Result<Self> {
        Ok(Self {
            collector: LogCollector::new(config, contract_decoder)?,
        })
    }
}

impl<ChainSpecT> edr_provider::Logger<ChainSpecT> for Logger<ChainSpecT>
where
    ChainSpecT: ProviderSpec<CurrentTime>,
{
    type BlockchainError = BlockchainErrorForChainSpec<ChainSpecT>;

    fn is_enabled(&self) -> bool {
        self.collector.config.enable
    }

    fn set_is_enabled(&mut self, is_enabled: bool) {
        self.collector.config.enable = is_enabled;
    }

    fn log_call(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &CallResult<ChainSpecT::HaltReason>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector.log_call(hardfork, transaction, result)?;

        Ok(())
    }

    fn log_estimate_gas_failure(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        failure: &EstimateGasFailure<ChainSpecT::HaltReason>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector
            .log_estimate_gas(hardfork, transaction, failure)?;

        Ok(())
    }

    fn log_interval_mined(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        mining_result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector
            .log_interval_mined(hardfork, mining_result)
            .map_err(Box::new)?;

        Ok(())
    }

    fn log_mined_block(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        mining_results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector.log_mined_blocks(hardfork, mining_results)?;

        Ok(())
    }

    fn log_send_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        mining_results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector
            .log_send_transaction(hardfork, transaction, mining_results)?;

        Ok(())
    }

    fn print_contract_decoding_error(
        &mut self,
        error: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.collector.log(
            "Contract decoder failed to be updated. Please report this to help us improve Hardhat.",
        );
        self.collector.print_empty_line()?;
        self.collector.log(error);
        Ok(())
    }

    fn print_method_logs(
        &mut self,
        method: &str,
        error: Option<&ProviderError<ChainSpecT>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(error) = error {
            self.collector.state = LoggingState::Empty;

            if matches!(error, ProviderError::UnsupportedMethod { .. }) {
                self.collector
                    .print::<false>(Color::Red.paint(error.to_string()))?;
            } else {
                self.collector.print::<false>(Color::Red.paint(method))?;
                self.collector.print_logs()?;

                if !matches!(error, ProviderError::TransactionFailed(_)) {
                    self.collector.print_empty_line()?;

                    let error_message = error.to_string();
                    self.collector
                        .try_indented(|logger| logger.print::<false>(&error_message))?;

                    if matches!(error, ProviderError::InvalidEip155TransactionChainId) {
                        self.collector.try_indented(|logger| {
                            logger.print::<false>(Color::Yellow.paint(
                                "If you are using MetaMask, you can learn how to fix this error here: https://hardhat.org/metamask-issue"
                            ))
                        })?;
                    }
                }

                self.collector.print_empty_line()?;
            }
        } else {
            self.collector.print_method(method)?;

            let printed = self.collector.print_logs()?;
            if printed {
                self.collector.print_empty_line()?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct CollapsedMethod {
    count: usize,
    method: String,
}

#[derive_where(Clone)]
struct LogCollector<ChainSpecT: ProviderSpec<CurrentTime>> {
    config: Config,
    contract_decoder: Arc<ContractDecoder>,
    indentation: usize,
    logs: Vec<LogLine>,
    state: LoggingState,
    title_length: usize,
    phantom: PhantomData<ChainSpecT>,
}

impl<ChainSpecT: ProviderSpec<CurrentTime>> LogCollector<ChainSpecT> {
    pub fn new(config: Config, contract_decoder: Arc<ContractDecoder>) -> napi::Result<Self> {
        Ok(Self {
            config,
            contract_decoder,
            indentation: 0,
            logs: Vec::new(),
            state: LoggingState::default(),
            title_length: 0,
            phantom: PhantomData,
        })
    }

    pub fn log_call(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &CallResult<ChainSpecT::HaltReason>,
    ) -> Result<(), LoggerError> {
        let CallResult {
            console_log_inputs,
            execution_result,
            trace,
        } = result;

        self.state = LoggingState::Empty;

        self.indented(|logger| {
            logger.log_contract_and_function_name::<true>(hardfork, trace);

            logger.log_with_title("From", format!("0x{:x}", transaction.caller()));
            if let Some(to) = transaction.kind().to() {
                logger.log_with_title("To", format!("0x{to:x}"));
            }
            if *transaction.value() > U256::ZERO {
                logger.log_with_title("Value", wei_to_human_readable(transaction.value()));
            }

            logger.log_console_log_messages(console_log_inputs)?;

            if let Some(transaction_failure) = TransactionFailure::from_execution_result::<
                ChainSpecT,
                CurrentTime,
            >(execution_result, None, trace)
            {
                logger.log_transaction_failure(&transaction_failure);
            }

            Ok(())
        })
    }

    pub fn log_estimate_gas(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &EstimateGasFailure<ChainSpecT::HaltReason>,
    ) -> Result<(), LoggerError> {
        let EstimateGasFailure {
            console_log_inputs,
            transaction_failure,
        } = result;

        self.state = LoggingState::Empty;

        self.indented(|logger| {
            logger.log_contract_and_function_name::<true>(
                hardfork,
                &transaction_failure.failure.solidity_trace,
            );

            logger.log_with_title("From", format!("0x{:x}", transaction.caller()));
            if let Some(to) = transaction.kind().to() {
                logger.log_with_title("To", format!("0x{to:x}"));
            }
            logger.log_with_title("Value", wei_to_human_readable(transaction.value()));

            logger.log_console_log_messages(console_log_inputs)?;

            logger.log_transaction_failure(&transaction_failure.failure);

            Ok(())
        })
    }

    fn log_transaction_failure(
        &mut self,
        failure: &edr_provider::TransactionFailure<ChainSpecT::HaltReason>,
    ) {
        let is_revert_error = matches!(
            failure.reason,
            edr_provider::TransactionFailureReason::Revert(_)
        );

        let error_type = if is_revert_error {
            "Error"
        } else {
            "TransactionExecutionError"
        };

        self.log_empty_line();
        self.log(format!("{error_type}: {failure}"));
    }

    pub fn log_mined_blocks(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        mining_results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), LoggerError> {
        let num_results = mining_results.len();
        for (idx, mining_result) in mining_results.iter().enumerate() {
            let state = std::mem::take(&mut self.state);
            let empty_blocks_range_start = state.into_hardhat_mining();

            if mining_result.block.transactions().is_empty() {
                self.log_hardhat_mined_empty_block(&mining_result.block, empty_blocks_range_start)?;

                let block_number = mining_result.block.header().number;
                self.state = LoggingState::HardhatMinining {
                    empty_blocks_range_start: Some(
                        empty_blocks_range_start.unwrap_or(block_number),
                    ),
                };
            } else {
                self.log_hardhat_mined_block(hardfork, mining_result)?;

                if idx < num_results - 1 {
                    self.log_empty_line();
                }
            }
        }

        Ok(())
    }

    pub fn log_interval_mined(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        mining_result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
    ) -> Result<(), LoggerError> {
        let block_header = mining_result.block.header();
        let block_number = block_header.number;

        if mining_result.block.transactions().is_empty() {
            let state = std::mem::take(&mut self.state);
            let empty_blocks_range_start = state.into_interval_mining();

            if let Some(empty_blocks_range_start) = empty_blocks_range_start {
                self.print::<true>(format!(
                    "Mined empty block range #{empty_blocks_range_start} to #{block_number}"
                ))?;
            } else {
                let base_fee = if let Some(base_fee) = block_header.base_fee_per_gas.as_ref() {
                    format!(" with base fee {base_fee}")
                } else {
                    String::new()
                };

                self.print::<false>(format!("Mined empty block #{block_number}{base_fee}"))?;
            }

            self.state = LoggingState::IntervalMining {
                empty_blocks_range_start: Some(
                    empty_blocks_range_start.unwrap_or(block_header.number),
                ),
            };
        } else {
            self.log_interval_mined_block(hardfork, mining_result)?;

            self.print::<false>(format!("Mined block #{block_number}"))?;

            let printed = self.print_logs()?;
            if printed {
                self.print_empty_line()?;
            }
        }

        Ok(())
    }

    pub fn log_send_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        mining_results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
    ) -> Result<(), LoggerError> {
        if !mining_results.is_empty() {
            self.state = LoggingState::Empty;

            let (sent_block_result, sent_transaction_result, sent_trace) = mining_results
                .iter()
                .find_map(|result| {
                    izip!(
                        result.block.transactions(),
                        result.transaction_results.iter(),
                        result.transaction_traces.iter()
                    )
                    .find(|(block_transaction, _, _)| {
                        *block_transaction.transaction_hash() == *transaction.transaction_hash()
                    })
                    .map(|(_, transaction_result, trace)| (result, transaction_result, trace))
                })
                .expect("Transaction result not found");

            if mining_results.len() > 1 {
                self.log_multiple_blocks_warning()?;
                self.log_auto_mined_block_results(
                    hardfork,
                    mining_results,
                    transaction.transaction_hash(),
                )?;
                self.log_currently_sent_transaction(
                    hardfork,
                    sent_block_result,
                    transaction,
                    sent_transaction_result,
                    sent_trace,
                )?;
            } else if let Some(result) = mining_results.first() {
                let transactions = result.block.transactions();
                if transactions.len() > 1 {
                    self.log_multiple_transactions_warning()?;
                    self.log_auto_mined_block_results(
                        hardfork,
                        mining_results,
                        transaction.transaction_hash(),
                    )?;
                    self.log_currently_sent_transaction(
                        hardfork,
                        sent_block_result,
                        transaction,
                        sent_transaction_result,
                        sent_trace,
                    )?;
                } else if let Some(transaction) = transactions.first() {
                    self.log_single_transaction_mining_result(hardfork, result, transaction)?;
                }
            }
        }

        Ok(())
    }

    fn format(&self, message: impl ToString) -> String {
        let message = message.to_string();

        if message.is_empty() {
            message
        } else {
            message
                .split('\n')
                .map(|line| format!("{:indent$}{line}", "", indent = self.indentation))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    fn indented(
        &mut self,
        display_fn: impl FnOnce(&mut Self) -> Result<(), LoggerError>,
    ) -> Result<(), LoggerError> {
        self.indentation += 2;
        let result = display_fn(self);
        self.indentation -= 2;

        // We need to return the result of the inner function after resetting the
        // indentation
        result
    }

    fn try_indented(
        &mut self,
        display_fn: impl FnOnce(&mut Self) -> Result<(), LoggerError>,
    ) -> Result<(), LoggerError> {
        self.indentation += 2;
        let result = display_fn(self);
        self.indentation -= 2;

        result
    }

    fn log(&mut self, message: impl ToString) {
        let formatted = self.format(message);

        self.logs.push(LogLine::Single(formatted));
    }

    fn log_auto_mined_block_results(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        results: &[DebugMineBlockResultForChainSpec<ChainSpecT>],
        sent_transaction_hash: &B256,
    ) -> Result<(), LoggerError> {
        for result in results {
            self.log_block_from_auto_mine(hardfork, result, sent_transaction_hash)?;
        }

        Ok(())
    }

    fn log_base_fee(&mut self, base_fee: Option<&U256>) {
        if let Some(base_fee) = base_fee {
            self.log(format!("Base fee: {base_fee}"));
        }
    }

    fn log_block_from_auto_mine(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
        transaction_hash_to_highlight: &edr_eth::B256,
    ) -> Result<(), LoggerError> {
        let DebugMineBlockResult {
            block,
            transaction_results,
            transaction_traces,
            console_log_inputs,
            ..
        } = result;

        let transactions = block.transactions();
        let num_transactions = transactions.len();

        debug_assert_eq!(num_transactions, transaction_results.len());
        debug_assert_eq!(num_transactions, transaction_traces.len());

        let block_header = block.header();

        self.indented(|logger| {
            logger.log_block_id(block);

            logger.indented(|logger| {
                logger.log_base_fee(block_header.base_fee_per_gas.as_ref());

                for (idx, transaction, result, trace) in izip!(
                    0..num_transactions,
                    transactions,
                    transaction_results,
                    transaction_traces
                ) {
                    let should_highlight_hash =
                        *transaction.transaction_hash() == *transaction_hash_to_highlight;
                    logger.log_block_transaction(
                        hardfork,
                        transaction,
                        result,
                        trace,
                        console_log_inputs,
                        should_highlight_hash,
                    )?;

                    logger.log_empty_line_between_transactions(idx, num_transactions);
                }

                Ok(())
            })?;

            Ok(())
        })?;

        self.log_empty_line();

        Ok(())
    }

    fn log_block_hash(&mut self, block: &ChainSpecT::Block) {
        let block_hash = block.block_hash();

        self.log(format!("Block: {block_hash}"));
    }

    fn log_block_id(&mut self, block: &ChainSpecT::Block) {
        let block_number = block.header().number;
        let block_hash = block.block_hash();

        self.log(format!("Block #{block_number}: {block_hash}"));
    }

    fn log_block_number(&mut self, block: &ChainSpecT::Block) {
        let block_number = block.header().number;

        self.log(format!("Mined block #{block_number}"));
    }

    /// Logs a transaction that's part of a block.
    fn log_block_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        transaction: &ChainSpecT::SignedTransaction,
        result: &ExecutionResult<ChainSpecT::HaltReason>,
        trace: &Trace<ChainSpecT::HaltReason>,
        console_log_inputs: &[Bytes],
        should_highlight_hash: bool,
    ) -> Result<(), LoggerError> {
        let transaction_hash = transaction.transaction_hash();
        if should_highlight_hash {
            self.log_with_title(
                "Transaction",
                Style::new().bold().paint(transaction_hash.to_string()),
            );
        } else {
            self.log_with_title("Transaction", transaction_hash.to_string());
        }

        self.indented(|logger| {
            logger.log_contract_and_function_name::<false>(hardfork, trace);
            logger.log_with_title("From", format!("0x{:x}", transaction.caller()));
            if let Some(to) = transaction.kind().to() {
                logger.log_with_title("To", format!("0x{to:x}"));
            }
            logger.log_with_title("Value", wei_to_human_readable(transaction.value()));
            logger.log_with_title(
                "Gas used",
                format!(
                    "{gas_used} of {gas_limit}",
                    gas_used = result.gas_used(),
                    gas_limit = transaction.gas_limit()
                ),
            );

            logger.log_console_log_messages(console_log_inputs)?;

            let transaction_failure = edr_provider::TransactionFailure::from_execution_result::<
                ChainSpecT,
                CurrentTime,
            >(result, Some(transaction_hash), trace);

            if let Some(transaction_failure) = transaction_failure {
                logger.log_transaction_failure(&transaction_failure);
            }

            Ok(())
        })?;

        Ok(())
    }

    fn log_console_log_messages(
        &mut self,
        console_log_inputs: &[Bytes],
    ) -> Result<(), LoggerError> {
        let console_log_inputs =
            (self.config.decode_console_log_inputs_fn)(console_log_inputs.to_vec());

        // This is a special case, as we always want to print the console.log messages.
        // The difference is how. If we have a logger, we should use that, so that logs
        // are printed in order. If we don't, we just print the messages here.
        if self.config.enable {
            if !console_log_inputs.is_empty() {
                self.log_empty_line();
                self.log("console.log:");

                self.indented(|logger| {
                    for input in console_log_inputs {
                        logger.log(input);
                    }

                    Ok(())
                })?;
            }
        } else {
            for input in console_log_inputs {
                (self.config.print_line_fn)(input, false)?;
            }
        }

        Ok(())
    }

    fn log_contract_and_function_name<const PRINT_INVALID_CONTRACT_WARNING: bool>(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        trace: &Trace<ChainSpecT::HaltReason>,
    ) {
        if let Some(TraceMessage::Before(before_message)) = trace.messages.first() {
            if let Some(to) = before_message.to {
                // Call
                let is_precompile = {
                    let precompiles = Precompiles::new(precompile::PrecompileSpecId::from_spec_id(
                        hardfork.into(),
                    ));
                    precompiles.contains(&to)
                };

                if is_precompile {
                    let precompile = u16::from_be_bytes([to[18], to[19]]);
                    self.log_with_title(
                        "Precompile call",
                        format!("<PrecompileContract {precompile}>"),
                    );
                } else {
                    let is_code_empty = before_message
                        .code
                        .as_ref()
                        .map_or(true, edr_eth::Bytecode::is_empty);

                    if is_code_empty {
                        if PRINT_INVALID_CONTRACT_WARNING {
                            self.log("WARNING: Calling an account which is not a contract");
                        }
                    } else {
                        let code = before_message
                            .code
                            .as_ref()
                            .map(edr_eth::Bytecode::original_bytes)
                            .expect("Call must be defined");

                        let ContractAndFunctionName {
                            contract_name,
                            function_name,
                        } = self
                            .contract_decoder
                            .get_contract_and_function_names_for_call(
                                &code,
                                Some(&before_message.data),
                            );

                        let function_name = function_name.expect("Function name must be defined");
                        self.log_with_title(
                            "Contract call",
                            if function_name.is_empty() {
                                contract_name
                            } else {
                                format!("{contract_name}#{function_name}")
                            },
                        );
                    }
                }
            } else {
                let result = if let Some(TraceMessage::After(AfterMessage {
                    execution_result,
                    ..
                })) = trace.messages.last()
                {
                    execution_result
                } else {
                    unreachable!("Before messages must have an after message")
                };

                // Create
                let ContractAndFunctionName { contract_name, .. } = self
                    .contract_decoder
                    .get_contract_and_function_names_for_call(&before_message.data, None);

                self.log_with_title("Contract deployment", contract_name);

                if let ExecutionResult::Success { output, .. } = result {
                    if let edr_eth::result::Output::Create(_, address) = output {
                        if let Some(deployed_address) = address {
                            self.log_with_title(
                                "Contract address",
                                format!("0x{deployed_address:x}"),
                            );
                        }
                    } else {
                        unreachable!("Create calls must return a Create output")
                    }
                }
            }
        }
    }

    fn log_empty_block(&mut self, block: &ChainSpecT::Block) {
        let block_header = block.header();
        let block_number = block_header.number;

        let base_fee = if let Some(base_fee) = block_header.base_fee_per_gas.as_ref() {
            format!(" with base fee {base_fee}")
        } else {
            String::new()
        };

        self.log(format!("Mined empty block #{block_number}{base_fee}",));
    }

    fn log_empty_line(&mut self) {
        self.log("");
    }

    fn log_empty_line_between_transactions(&mut self, idx: usize, num_transactions: usize) {
        if num_transactions > 1 && idx < num_transactions - 1 {
            self.log_empty_line();
        }
    }

    fn log_hardhat_mined_empty_block(
        &mut self,
        block: &ChainSpecT::Block,
        empty_blocks_range_start: Option<u64>,
    ) -> Result<(), LoggerError> {
        self.indented(|logger| {
            if let Some(empty_blocks_range_start) = empty_blocks_range_start {
                logger.replace_last_log_line(format!(
                    "Mined empty block range #{empty_blocks_range_start} to #{block_number}",
                    block_number = block.header().number
                ));
            } else {
                logger.log_empty_block(block);
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Logs the result of interval mining a block.
    fn log_interval_mined_block(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
    ) -> Result<(), LoggerError> {
        let DebugMineBlockResult {
            block,
            transaction_results,
            transaction_traces,
            console_log_inputs,
            ..
        } = result;

        let transactions = block.transactions();
        let num_transactions = transactions.len();

        debug_assert_eq!(num_transactions, transaction_results.len());
        debug_assert_eq!(num_transactions, transaction_traces.len());

        let block_header = block.header();

        self.indented(|logger| {
            logger.log_block_hash(block);

            logger.indented(|logger| {
                logger.log_base_fee(block_header.base_fee_per_gas.as_ref());

                for (idx, transaction, result, trace) in izip!(
                    0..num_transactions,
                    transactions,
                    transaction_results,
                    transaction_traces
                ) {
                    logger.log_block_transaction(
                        hardfork,
                        transaction,
                        result,
                        trace,
                        console_log_inputs,
                        false,
                    )?;

                    logger.log_empty_line_between_transactions(idx, num_transactions);
                }

                Ok(())
            })
        })
    }

    fn log_hardhat_mined_block(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
    ) -> Result<(), LoggerError> {
        let DebugMineBlockResult {
            block,
            transaction_results,
            transaction_traces,
            console_log_inputs,
            ..
        } = result;

        let transactions = block.transactions();
        let num_transactions = transactions.len();

        debug_assert_eq!(num_transactions, transaction_results.len());
        debug_assert_eq!(num_transactions, transaction_traces.len());

        self.indented(|logger| {
            if transactions.is_empty() {
                logger.log_empty_block(block);
            } else {
                logger.log_block_number(block);

                logger.indented(|logger| {
                    logger.log_block_hash(block);

                    logger.indented(|logger| {
                        logger.log_base_fee(block.header().base_fee_per_gas.as_ref());

                        for (idx, transaction, result, trace) in izip!(
                            0..num_transactions,
                            transactions,
                            transaction_results,
                            transaction_traces
                        ) {
                            logger.log_block_transaction(
                                hardfork,
                                transaction,
                                result,
                                trace,
                                console_log_inputs,
                                false,
                            )?;

                            logger.log_empty_line_between_transactions(idx, num_transactions);
                        }

                        Ok(())
                    })
                })?;
            }

            Ok(())
        })
    }

    /// Logs a warning about multiple blocks being mined.
    fn log_multiple_blocks_warning(&mut self) -> Result<(), LoggerError> {
        self.indented(|logger| {
            logger
                .log("There were other pending transactions. More than one block had to be mined:");

            Ok(())
        })?;
        self.log_empty_line();

        Ok(())
    }

    /// Logs a warning about multiple transactions being mined.
    fn log_multiple_transactions_warning(&mut self) -> Result<(), LoggerError> {
        self.indented(|logger| {
            logger.log("There were other pending transactions mined in the same block:");

            Ok(())
        })?;
        self.log_empty_line();

        Ok(())
    }

    fn log_with_title(&mut self, title: impl Into<String>, message: impl Display) {
        // repeat whitespace self.indentation times and concatenate with title
        let title = format!("{:indent$}{}", "", title.into(), indent = self.indentation);
        if title.len() > self.title_length {
            self.title_length = title.len();
        }

        let message = format!("{message}");
        self.logs.push(LogLine::WithTitle(title, message));
    }

    fn log_currently_sent_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        block_result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
        transaction: &ChainSpecT::SignedTransaction,
        transaction_result: &ExecutionResult<ChainSpecT::HaltReason>,
        trace: &Trace<ChainSpecT::HaltReason>,
    ) -> Result<(), LoggerError> {
        self.indented(|logger| {
            logger.log("Currently sent transaction:");
            logger.log("");

            Ok(())
        })?;

        self.log_transaction(
            hardfork,
            block_result,
            transaction,
            transaction_result,
            trace,
        )?;

        Ok(())
    }

    fn log_single_transaction_mining_result(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
        transaction: &ChainSpecT::SignedTransaction,
    ) -> Result<(), LoggerError> {
        let trace = result
            .transaction_traces
            .first()
            .expect("A transaction exists, so the trace must exist as well.");

        let transaction_result = result
            .transaction_results
            .first()
            .expect("A transaction exists, so the result must exist as well.");

        self.log_transaction(hardfork, result, transaction, transaction_result, trace)?;

        Ok(())
    }

    fn log_transaction(
        &mut self,
        hardfork: ChainSpecT::Hardfork,
        block_result: &DebugMineBlockResultForChainSpec<ChainSpecT>,
        transaction: &ChainSpecT::SignedTransaction,
        transaction_result: &ExecutionResult<ChainSpecT::HaltReason>,
        trace: &Trace<ChainSpecT::HaltReason>,
    ) -> Result<(), LoggerError> {
        self.indented(|logger| {
            logger.log_contract_and_function_name::<false>(hardfork, trace);

            let transaction_hash = transaction.transaction_hash();
            logger.log_with_title("Transaction", transaction_hash);

            logger.log_with_title("From", format!("0x{:x}", transaction.caller()));
            if let Some(to) = transaction.kind().to() {
                logger.log_with_title("To", format!("0x{to:x}"));
            }
            logger.log_with_title("Value", wei_to_human_readable(transaction.value()));
            logger.log_with_title(
                "Gas used",
                format!(
                    "{gas_used} of {gas_limit}",
                    gas_used = transaction_result.gas_used(),
                    gas_limit = transaction.gas_limit()
                ),
            );

            let block_number = block_result.block.header().number;
            logger.log_with_title(
                format!("Block #{block_number}"),
                block_result.block.block_hash(),
            );

            logger.log_console_log_messages(&block_result.console_log_inputs)?;

            let transaction_failure = edr_provider::TransactionFailure::from_execution_result::<
                ChainSpecT,
                CurrentTime,
            >(
                transaction_result, Some(transaction_hash), trace
            );

            if let Some(transaction_failure) = transaction_failure {
                logger.log_transaction_failure(&transaction_failure);
            }

            Ok(())
        })
    }

    fn print<const REPLACE: bool>(&mut self, message: impl ToString) -> Result<(), LoggerError> {
        if !self.config.enable {
            return Ok(());
        }

        let formatted = self.format(message);
        (self.config.print_line_fn)(formatted, REPLACE)
    }

    fn print_empty_line(&mut self) -> Result<(), LoggerError> {
        self.print::<false>("")
    }

    fn print_logs(&mut self) -> Result<bool, LoggerError> {
        let logs = std::mem::take(&mut self.logs);
        if logs.is_empty() {
            return Ok(false);
        }

        for log in logs {
            let line = match log {
                LogLine::Single(message) => message,
                LogLine::WithTitle(title, message) => {
                    let title = format!("{title}:");
                    format!("{title:indent$} {message}", indent = self.title_length + 1)
                }
            };

            self.print::<false>(line)?;
        }

        Ok(true)
    }

    fn print_method(&mut self, method: &str) -> Result<(), LoggerError> {
        if let Some(collapsed_method) = self.collapsed_method(method) {
            collapsed_method.count += 1;

            let line = format!("{method} ({count})", count = collapsed_method.count);
            self.print::<true>(Color::Green.paint(line))
        } else {
            self.state = LoggingState::CollapsingMethod(CollapsedMethod {
                count: 1,
                method: method.to_string(),
            });

            self.print::<false>(Color::Green.paint(method))
        }
    }

    /// Retrieves the collapsed method with the provided name, if it exists.
    fn collapsed_method(&mut self, method: &str) -> Option<&mut CollapsedMethod> {
        if let LoggingState::CollapsingMethod(collapsed_method) = &mut self.state {
            if collapsed_method.method == method {
                return Some(collapsed_method);
            }
        }

        None
    }

    fn replace_last_log_line(&mut self, message: impl ToString) {
        let formatted = self.format(message);

        *self.logs.last_mut().expect("There must be a log line") = LogLine::Single(formatted);
    }
}

fn wei_to_human_readable(wei: &U256) -> String {
    if *wei == U256::ZERO {
        "0 ETH".to_string()
    } else if *wei < U256::from(100_000u64) {
        format!("{wei} wei")
    } else if *wei < U256::from(100_000_000_000_000u64) {
        let mut decimal = to_decimal_string(wei, 9);
        decimal.push_str(" gwei");
        decimal
    } else {
        let mut decimal = to_decimal_string(wei, 18);
        decimal.push_str(" ETH");
        decimal
    }
}

/// Converts the provided `value` to a decimal string after dividing it by
/// `10^exponent`. The returned string will have at most `MAX_DECIMALS`
/// decimals.
fn to_decimal_string(value: &U256, exponent: u8) -> String {
    const MAX_DECIMALS: u8 = 4;

    let (integer, remainder) = value.div_rem(U256::from(10).pow(U256::from(exponent)));
    let decimal = remainder / U256::from(10).pow(U256::from(exponent - MAX_DECIMALS));

    // Remove trailing zeros
    let decimal = decimal.to_string().trim_end_matches('0').to_string();

    format!("{integer}.{decimal}")
}
