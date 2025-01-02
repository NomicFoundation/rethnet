use std::fmt::{Debug, Formatter};

use napi::{
    bindgen_prelude::{BigInt, Buffer, Either3},
    Either,
};
use napi_derive::napi;

use crate::solidity_tests::artifact::ArtifactId;

/// See [edr_solidity_tests::result::SuiteResult]
#[napi(object)]
#[derive(Debug, Clone)]
pub struct SuiteResult {
    /// The artifact id can be used to match input to result in the progress
    /// callback
    #[napi(readonly)]
    pub id: ArtifactId,
    /// See [edr_solidity_tests::result::SuiteResult::duration]
    #[napi(readonly)]
    pub duration_ms: BigInt,
    /// See [edr_solidity_tests::result::SuiteResult::test_results]
    #[napi(readonly)]
    pub test_results: Vec<TestResult>,
    /// See [edr_solidity_tests::result::SuiteResult::warnings]
    #[napi(readonly)]
    pub warnings: Vec<String>,
}

impl
    From<(
        edr_solidity_tests::contracts::ArtifactId,
        edr_solidity_tests::result::SuiteResult,
    )> for SuiteResult
{
    fn from(
        (id, suite_result): (
            edr_solidity_tests::contracts::ArtifactId,
            edr_solidity_tests::result::SuiteResult,
        ),
    ) -> Self {
        Self {
            id: id.into(),
            duration_ms: BigInt::from(suite_result.duration.as_millis()),
            test_results: suite_result
                .test_results
                .into_iter()
                .map(Into::into)
                .collect(),
            warnings: suite_result.warnings,
        }
    }
}

/// See [edr_solidity_tests::result::TestResult]
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TestResult {
    /// The name of the test.
    #[napi(readonly)]
    pub name: String,
    /// See [edr_solidity_tests::result::TestResult::status]
    #[napi(readonly)]
    pub status: TestStatus,
    /// See [edr_solidity_tests::result::TestResult::reason]
    #[napi(readonly)]
    pub reason: Option<String>,
    /// See [edr_solidity_tests::result::TestResult::counterexample]
    #[napi(readonly)]
    pub counterexample: Option<Either<BaseCounterExample, Vec<BaseCounterExample>>>,
    /// See [edr_solidity_tests::result::TestResult::decoded_logs]
    #[napi(readonly)]
    pub decoded_logs: Vec<String>,
    /// See [edr_solidity_tests::result::TestResult::kind]
    #[napi(readonly)]
    pub kind: Either3<StandardTestKind, FuzzTestKind, InvariantTestKind>,
    /// See [edr_solidity_tests::result::TestResult::duration]
    #[napi(readonly)]
    pub duration_ms: BigInt,
}

impl From<(String, edr_solidity_tests::result::TestResult)> for TestResult {
    fn from((name, test_result): (String, edr_solidity_tests::result::TestResult)) -> Self {
        Self {
            name,
            status: test_result.status.into(),
            reason: test_result.reason,
            counterexample: test_result
                .counterexample
                .map(|counterexample| match counterexample {
                    edr_solidity_tests::fuzz::CounterExample::Single(counterexample) => {
                        Either::A(BaseCounterExample::from(counterexample))
                    }
                    edr_solidity_tests::fuzz::CounterExample::Sequence(counterexamples) => {
                        Either::B(
                            counterexamples
                                .into_iter()
                                .map(BaseCounterExample::from)
                                .collect(),
                        )
                    }
                }),
            decoded_logs: test_result.decoded_logs,
            kind: match test_result.kind {
                edr_solidity_tests::result::TestKind::Standard(gas_consumed) => {
                    Either3::A(StandardTestKind {
                        consumed_gas: BigInt::from(gas_consumed),
                    })
                }
                edr_solidity_tests::result::TestKind::Fuzz {
                    runs,
                    mean_gas,
                    median_gas,
                } => Either3::B(FuzzTestKind {
                    // usize as u64 is always safe
                    runs: BigInt::from(runs as u64),
                    mean_gas: BigInt::from(mean_gas),
                    median_gas: BigInt::from(median_gas),
                }),
                edr_solidity_tests::result::TestKind::Invariant {
                    runs,
                    calls,
                    reverts,
                } => Either3::C(InvariantTestKind {
                    // usize as u64 is always safe
                    runs: BigInt::from(runs as u64),
                    calls: BigInt::from(calls as u64),
                    reverts: BigInt::from(reverts as u64),
                }),
            },
            duration_ms: BigInt::from(test_result.duration.as_millis()),
        }
    }
}

#[derive(Debug)]
#[napi(string_enum)]
#[doc = "The result of a test execution."]
pub enum TestStatus {
    #[doc = "Test success"]
    Success,
    #[doc = "Test failure"]
    Failure,
    #[doc = "Test skipped"]
    Skipped,
}

impl From<edr_solidity_tests::result::TestStatus> for TestStatus {
    fn from(value: edr_solidity_tests::result::TestStatus) -> Self {
        match value {
            edr_solidity_tests::result::TestStatus::Success => Self::Success,
            edr_solidity_tests::result::TestStatus::Failure => Self::Failure,
            edr_solidity_tests::result::TestStatus::Skipped => Self::Skipped,
        }
    }
}

/// See [edr_solidity_tests::result::TestKind::Standard]
#[napi(object)]
#[derive(Debug, Clone)]
pub struct StandardTestKind {
    /// The gas consumed by the test.
    #[napi(readonly)]
    pub consumed_gas: BigInt,
}

/// See [edr_solidity_tests::result::TestKind::Fuzz]
#[napi(object)]
#[derive(Debug, Clone)]
pub struct FuzzTestKind {
    /// See [edr_solidity_tests::result::TestKind::Fuzz]
    #[napi(readonly)]
    pub runs: BigInt,
    /// See [edr_solidity_tests::result::TestKind::Fuzz]
    #[napi(readonly)]
    pub mean_gas: BigInt,
    /// See [edr_solidity_tests::result::TestKind::Fuzz]
    #[napi(readonly)]
    pub median_gas: BigInt,
}

/// See [edr_solidity_tests::fuzz::FuzzCase]
#[napi(object)]
#[derive(Clone)]
pub struct FuzzCase {
    /// The calldata used for this fuzz test
    #[napi(readonly)]
    pub calldata: Buffer,
    /// Consumed gas
    #[napi(readonly)]
    pub gas: BigInt,
    /// The initial gas stipend for the transaction
    #[napi(readonly)]
    pub stipend: BigInt,
}

impl Debug for FuzzCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuzzCase")
            .field("gas", &self.gas)
            .field("stipend", &self.stipend)
            .finish()
    }
}

/// See [edr_solidity_tests::result::TestKind::Invariant]
#[napi(object)]
#[derive(Debug, Clone)]
pub struct InvariantTestKind {
    /// See [edr_solidity_tests::result::TestKind::Invariant]
    #[napi(readonly)]
    pub runs: BigInt,
    /// See [edr_solidity_tests::result::TestKind::Invariant]
    #[napi(readonly)]
    pub calls: BigInt,
    /// See [edr_solidity_tests::result::TestKind::Invariant]
    #[napi(readonly)]
    pub reverts: BigInt,
}

/// See [edr_solidity_tests::fuzz::BaseCounterExample]
#[napi(object)]
#[derive(Clone)]
pub struct BaseCounterExample {
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::sender]
    #[napi(readonly)]
    pub sender: Option<Buffer>,
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::addr]
    #[napi(readonly)]
    pub address: Option<Buffer>,
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::calldata]
    #[napi(readonly)]
    pub calldata: Buffer,
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::contract_name]
    #[napi(readonly)]
    pub contract_name: Option<String>,
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::signature]
    #[napi(readonly)]
    pub signature: Option<String>,
    /// See [edr_solidity_tests::fuzz::BaseCounterExample::args]
    #[napi(readonly)]
    pub args: Option<String>,
}

impl Debug for BaseCounterExample {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseCounterExample")
            .field("contract_name", &self.contract_name)
            .field("signature", &self.signature)
            .field("args", &self.args)
            .finish()
    }
}

impl From<edr_solidity_tests::fuzz::BaseCounterExample> for BaseCounterExample {
    fn from(value: edr_solidity_tests::fuzz::BaseCounterExample) -> Self {
        Self {
            sender: value.sender.map(|sender| Buffer::from(sender.as_slice())),
            address: value.addr.map(|address| Buffer::from(address.as_slice())),
            calldata: Buffer::from(value.calldata.as_ref()),
            contract_name: value.contract_name,
            signature: value.signature,
            args: value.args,
        }
    }
}
