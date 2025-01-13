use std::sync::Arc;

use edr_eth::HashMap;
use edr_napi_core::provider::{self, SyncProviderFactory};
use edr_solidity::contract_decoder::ContractDecoder;
use napi::{
    tokio::{runtime, sync::Mutex as AsyncMutex},
    Env, JsObject,
};
use napi_derive::napi;
use tracing_subscriber::{prelude::*, EnvFilter, Registry};

use crate::{
    config::ProviderConfig,
    logger::LoggerConfig,
    provider::{Provider, ProviderFactory},
    subscription::SubscriptionConfig,
};

#[napi]
pub struct EdrContext {
    inner: Arc<AsyncMutex<Context>>,
}

#[napi]
impl EdrContext {
    #[doc = "Creates a new [`EdrContext`] instance. Should only be called once!"]
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        let context = Context::new()?;

        Ok(Self {
            inner: Arc::new(AsyncMutex::new(context)),
        })
    }

    #[doc = "Constructs a new provider with the provided configuration."]
    #[napi(ts_return_type = "Promise<Provider>")]
    pub fn create_provider(
        &self,
        env: Env,
        chain_type: String,
        provider_config: ProviderConfig,
        logger_config: LoggerConfig,
        subscription_config: SubscriptionConfig,
        tracing_config: serde_json::Value,
    ) -> napi::Result<JsObject> {
        let provider_config = edr_napi_core::provider::Config::try_from(provider_config)?;
        let logger_config = logger_config.resolve(&env)?;

        // TODO: https://github.com/NomicFoundation/edr/issues/760
        let build_info_config: edr_solidity::contract_decoder::BuildInfoConfig =
            serde_json::from_value(tracing_config)?;

        let contract_decoder = ContractDecoder::new(&build_info_config).map_or_else(
            |error| Err(napi::Error::from_reason(error.to_string())),
            |contract_decoder| Ok(Arc::new(contract_decoder)),
        )?;

        #[cfg(feature = "scenarios")]
        let scenario_file =
            runtime::Handle::current().block_on(crate::scenarios::scenario_file(
                chain_type.clone(),
                provider_config.clone(),
                logger_config.enable,
            ))?;

        let runtime = runtime::Handle::current();
        let builder = {
            // TODO: https://github.com/NomicFoundation/edr/issues/760
            // TODO: Don't block the JS event loop
            let context = runtime.block_on(async { self.inner.lock().await });

            context.create_provider_builder(
                &env,
                &chain_type,
                provider_config,
                logger_config,
                subscription_config.into(),
                &contract_decoder,
            )?
        };

        let (deferred, promise) = env.create_deferred()?;
        runtime.clone().spawn_blocking(move || {
            let result = builder.build(runtime.clone()).map(|provider| {
                Provider::new(
                    provider,
                    runtime,
                    contract_decoder,
                    #[cfg(feature = "scenarios")]
                    scenario_file,
                )
            });

            deferred.resolve(|_env| result);
        });

        Ok(promise)
    }

    #[doc = "Registers a new provider factory for the provided chain type."]
    #[napi]
    pub async fn register_provider_factory(
        &self,
        chain_type: String,
        factory: &ProviderFactory,
    ) -> napi::Result<()> {
        let mut context = self.inner.lock().await;
        context.register_provider_factory(chain_type, factory.as_inner().clone());
        Ok(())
    }
}

pub struct Context {
    provider_factories: HashMap<String, Arc<dyn SyncProviderFactory>>,
    #[cfg(feature = "tracing")]
    _tracing_write_guard: tracing_flame::FlushGuard<std::io::BufWriter<std::fs::File>>,
}

impl Context {
    /// Creates a new [`Context`] instance. Should only be called once!
    pub fn new() -> napi::Result<Self> {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(true)
            .with_target(false)
            .with_level(true)
            .with_filter(EnvFilter::from_default_env());

        let subscriber = Registry::default().with(fmt_layer);

        #[cfg(feature = "tracing")]
        let (flame_layer, guard) = {
            let (flame_layer, guard) = tracing_flame::FlameLayer::with_file("tracing.folded")
                .map_err(|err| {
                    napi::Error::new(
                        napi::Status::GenericFailure,
                        format!("Failed to create tracing.folded file with error: {err:?}"),
                    )
                })?;

            let flame_layer = flame_layer.with_empty_samples(false);
            (flame_layer, guard)
        };

        #[cfg(feature = "tracing")]
        let subscriber = subscriber.with(flame_layer);

        if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
            println!(
                "Failed to set global tracing subscriber with error: {error}\n\
                Please only initialize EdrContext once per process to avoid this error."
            );
        }

        Ok(Self {
            provider_factories: HashMap::new(),
            #[cfg(feature = "tracing")]
            _tracing_write_guard: guard,
        })
    }

    /// Registers a new provider factory for the provided chain type.
    pub fn register_provider_factory(
        &mut self,
        chain_type: String,
        factory: Arc<dyn SyncProviderFactory>,
    ) {
        self.provider_factories.insert(chain_type, factory);
    }

    /// Tries to create a new provider for the provided chain type and
    /// configuration.
    pub fn create_provider_builder(
        &self,
        env: &napi::Env,
        chain_type: &str,
        provider_config: edr_napi_core::provider::Config,
        logger_config: edr_napi_core::logger::Config,
        subscription_config: edr_napi_core::subscription::Config,
        contract_decoder: &Arc<ContractDecoder>,
    ) -> napi::Result<Box<dyn provider::Builder>> {
        if let Some(factory) = self.provider_factories.get(chain_type) {
            factory.create_provider_builder(
                env,
                provider_config,
                logger_config,
                subscription_config,
                contract_decoder.clone(),
            )
        } else {
            Err(napi::Error::new(
                napi::Status::GenericFailure,
                "Provider for provided chain type does not exist",
            ))
        }
    }
}
