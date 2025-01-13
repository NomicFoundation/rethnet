use std::sync::Arc;

use edr_eth::{
    l1::L1ChainSpec,
    result::{HaltReason, InvalidTransaction},
    spec::HaltReasonTrait,
    transaction::{IsEip155, IsEip4844, TransactionMut, TransactionType, TransactionValidation},
};
use edr_evm::trace::Trace;
use edr_generic::GenericChainSpec;
use edr_provider::{time::CurrentTime, ProviderError, ResponseWithTraces, SyncProviderSpec};
use edr_rpc_client::jsonrpc;
use edr_solidity::contract_decoder::ContractDecoder;
use napi::{Either, Status};

pub type ResponseData = Either<String, serde_json::Value>;

pub struct Response<HaltReasonT: HaltReasonTrait> {
    // N-API is known to be slow when marshalling `serde_json::Value`s, so we try to return a
    // `String`. If the object is too large to be represented as a `String`, we return a `Buffer`
    // instead.
    pub data: ResponseData,
    /// When a transaction fails to execute, the provider returns a trace of the
    /// transaction.
    ///
    /// Only present for L1 Ethereum chains.
    pub solidity_trace: Option<SolidityTraceData<HaltReasonT>>,
    /// This may contain zero or more traces, depending on the (batch) request
    ///
    /// Always empty for non-L1 Ethereum chains.
    pub traces: Vec<Arc<Trace<HaltReasonT>>>,
}

impl<HaltReasonT: HaltReasonTrait> From<String> for Response<HaltReasonT> {
    fn from(value: String) -> Self {
        Response {
            data: Either::A(value),
            solidity_trace: None,
            traces: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct SolidityTraceData<HaltReasonT: HaltReasonTrait> {
    pub trace: Arc<Trace<HaltReasonT>>,
    pub contract_decoder: Arc<ContractDecoder>,
}

/// Trait for a defining a chain's associated type in the N-API.
pub trait SyncNapiSpec:
    SyncProviderSpec<
    CurrentTime,
    BlockEnv: Clone + Default,
    PooledTransaction: IsEip155,
    SignedTransaction: Default
                           + TransactionMut
                           + TransactionType<Type: IsEip4844>
                           + TransactionValidation<
        ValidationError: From<InvalidTransaction> + PartialEq,
    >,
>
{
    /// The string type identifier of the chain.
    const CHAIN_TYPE: &'static str;

    /// Casts a response with traces into a `Response`.
    ///
    /// This is implemented as an associated function to avoid problems when
    /// implementing type conversions for third-party types.
    fn cast_response(
        response: Result<ResponseWithTraces<Self::HaltReason>, ProviderError<Self>>,
        contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<Response<HaltReason>>;
}

impl SyncNapiSpec for L1ChainSpec {
    const CHAIN_TYPE: &'static str = "L1";

    fn cast_response(
        mut response: Result<ResponseWithTraces<Self::HaltReason>, ProviderError<Self>>,
        contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<Response<HaltReason>> {
        // We can take the solidity trace as it won't be used for anything else
        let solidity_trace = response.as_mut().err().and_then(|error| {
            if let edr_provider::ProviderError::TransactionFailed(failure) = error {
                if matches!(
                    failure.failure.reason,
                    edr_provider::TransactionFailureReason::OutOfGas(_)
                ) {
                    None
                } else {
                    Some(Arc::new(std::mem::take(
                        &mut failure.failure.solidity_trace,
                    )))
                }
            } else {
                None
            }
        });

        // We can take the traces as they won't be used for anything else
        let traces = match &mut response {
            Ok(response) => std::mem::take(&mut response.traces),
            Err(edr_provider::ProviderError::TransactionFailed(failure)) => {
                std::mem::take(&mut failure.traces)
            }
            Err(_) => Vec::new(),
        };

        let response = jsonrpc::ResponseData::from(response.map(|response| response.result));

        marshal_response_data(response).map(|data| {
            let solidity_trace = solidity_trace.map(|solidity_trace| SolidityTraceData {
                trace: solidity_trace,
                contract_decoder,
            });

            Response {
                solidity_trace,
                data,
                traces: traces.into_iter().map(Arc::new).collect(),
            }
        })
    }
}

impl SyncNapiSpec for GenericChainSpec {
    const CHAIN_TYPE: &'static str = "generic";

    fn cast_response(
        mut response: Result<ResponseWithTraces<Self::HaltReason>, ProviderError<Self>>,
        contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<Response<HaltReason>> {
        // We can take the solidity trace as it won't be used for anything else
        let solidity_trace = response.as_mut().err().and_then(|error| {
            if let edr_provider::ProviderError::TransactionFailed(failure) = error {
                if matches!(
                    failure.failure.reason,
                    edr_provider::TransactionFailureReason::OutOfGas(_)
                ) {
                    None
                } else {
                    Some(Arc::new(std::mem::take(
                        &mut failure.failure.solidity_trace,
                    )))
                }
            } else {
                None
            }
        });

        // We can take the traces as they won't be used for anything else
        let traces = match &mut response {
            Ok(response) => std::mem::take(&mut response.traces),
            Err(edr_provider::ProviderError::TransactionFailed(failure)) => {
                std::mem::take(&mut failure.traces)
            }
            Err(_) => Vec::new(),
        };

        let response = jsonrpc::ResponseData::from(response.map(|response| response.result));

        marshal_response_data(response).map(|data| {
            let solidity_trace = solidity_trace.map(|solidity_trace| SolidityTraceData {
                trace: solidity_trace,
                contract_decoder,
            });

            Response {
                data,
                solidity_trace,
                traces: traces.into_iter().map(Arc::new).collect(),
            }
        })
    }
}

/// Marshals a JSON-RPC response data into a `ResponseData`, taking into account
/// large responses.
pub fn marshal_response_data(
    response: jsonrpc::ResponseData<serde_json::Value>,
) -> napi::Result<ResponseData> {
    serde_json::to_string(&response)
        .and_then(|json| {
            // We experimentally determined that 500_000_000 was the maximum string length
            // that can be returned without causing the error:
            //
            // > Failed to convert rust `String` into napi `string`
            //
            // To be safe, we're limiting string lengths to half of that.
            const MAX_STRING_LENGTH: usize = 250_000_000;

            if json.len() <= MAX_STRING_LENGTH {
                Ok(Either::A(json))
            } else {
                serde_json::to_value(response).map(Either::B)
            }
        })
        .map_err(|error| napi::Error::new(Status::GenericFailure, error.to_string()))
}
