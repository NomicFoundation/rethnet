mod builder;
mod config;
mod factory;

use std::{str::FromStr as _, sync::Arc};

use edr_eth::result::HaltReason;
use edr_provider::{InvalidRequestReason, SyncCallOverride};
use edr_rpc_client::jsonrpc;
use edr_solidity::contract_decoder::ContractDecoder;

pub use self::{
    builder::{Builder, ProviderBuilder},
    config::{Config, HardforkActivation},
    factory::SyncProviderFactory,
};
use crate::spec::{Response, SyncNapiSpec};

/// Trait for a synchronous N-API provider that can be used for dynamic trait
/// objects.
pub trait SyncProvider: Send + Sync {
    /// Blocking method to handle a request.
    fn handle_request(
        &self,
        request: String,
        contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<Response<HaltReason>>;

    /// Set to `true` to make the traces returned with `eth_call`,
    /// `eth_estimateGas`, `eth_sendRawTransaction`, `eth_sendTransaction`,
    /// `evm_mine`, `hardhat_mine` include the full stack and memory. Set to
    /// `false` to disable this.
    fn set_call_override_callback(&self, call_override_callback: Arc<dyn SyncCallOverride>);

    /// Set the verbose tracing flag to the provided value.
    fn set_verbose_tracing(&self, enabled: bool);
}

impl<ChainSpecT: SyncNapiSpec> SyncProvider for edr_provider::Provider<ChainSpecT> {
    fn handle_request(
        &self,
        request: String,
        contract_decoder: Arc<ContractDecoder>,
    ) -> napi::Result<Response<HaltReason>> {
        let request = match serde_json::from_str(&request) {
            Ok(request) => request,
            Err(error) => {
                let message = error.to_string();

                let request = serde_json::Value::from_str(&request).ok();
                let method_name = request
                    .as_ref()
                    .and_then(|request| request.get("method"))
                    .and_then(serde_json::Value::as_str);

                let reason = InvalidRequestReason::new(method_name, &message);

                // HACK: We need to log failed deserialization attempts when they concern input
                // validation.
                if let Some((method_name, provider_error)) = reason.provider_error() {
                    // Ignore potential failure of logging, as returning the original error is more
                    // important
                    let _result = self.log_failed_deserialization(method_name, &provider_error);
                }

                let response = jsonrpc::ResponseData::<()>::Error {
                    error: jsonrpc::Error {
                        code: reason.error_code(),
                        message: reason.error_message(),
                        data: request,
                    },
                };

                return serde_json::to_string(&response)
                    .map_err(|error| {
                        napi::Error::new(
                            napi::Status::Unknown,
                            format!("Failed to serialize response due to: {error}"),
                        )
                    })
                    .map(Response::from);
            }
        };

        let response = edr_provider::Provider::handle_request(self, request);

        ChainSpecT::cast_response(response, contract_decoder)
    }

    fn set_call_override_callback(&self, call_override_callback: Arc<dyn SyncCallOverride>) {
        self.set_call_override_callback(Some(call_override_callback));
    }

    fn set_verbose_tracing(&self, enabled: bool) {
        self.set_verbose_tracing(enabled);
    }
}
