use core::fmt::Debug;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::Context;
use derive_where::derive_where;
use edr_eth::l1;
use edr_evm::{blockchain::BlockchainErrorForChainSpec, spec::RuntimeSpec};
use edr_generic::GenericChainSpec;
use edr_napi_core::spec::SyncNapiSpec;
use edr_provider::{time::CurrentTime, Logger, ProviderError, ProviderRequest};
use edr_rpc_eth::jsonrpc;
use edr_solidity::contract_decoder::ContractDecoder;
use flate2::bufread::GzDecoder;
use indicatif::ProgressBar;
use serde::Deserialize;
use tokio::{runtime, task};
#[cfg(feature = "tracing")]
use tracing_subscriber::{prelude::*, Registry};

#[derive(Clone, Debug, Deserialize)]
struct ScenarioConfig {
    chain_type: Option<String>,
    provider_config: edr_napi_core::provider::Config,
    logger_enabled: bool,
}

pub async fn execute(scenario_path: &Path, max_count: Option<usize>) -> anyhow::Result<()> {
    let (config, requests) = load_requests(scenario_path).await?;

    if config.logger_enabled {
        anyhow::bail!("This scenario expects logging, but logging is not yet implemented")
    }

    let provider_config = edr_provider::ProviderConfig::<l1::SpecId>::from(config.provider_config);

    let logger = Box::<DisabledLogger<GenericChainSpec>>::default();
    let subscription_callback = Box::new(|_| ());

    #[cfg(feature = "tracing")]
    let _flame_guard = {
        let (flame_layer, guard) = tracing_flame::FlameLayer::with_file("tracing.folded").unwrap();

        let flame_layer = flame_layer.with_empty_samples(false);
        let subscriber = Registry::default().with(flame_layer);

        tracing::subscriber::set_global_default(subscriber)
            .expect("Could not set global default tracing subscriber");

        guard
    };

    println!("Executing requests");

    let start = Instant::now();
    // Matches how `edr_napi` constructs and invokes the provider.
    let provider = task::spawn_blocking(move || {
        edr_provider::Provider::new(
            runtime::Handle::current(),
            logger,
            subscription_callback,
            provider_config,
            Arc::new(ContractDecoder::default()),
            CurrentTime,
        )
    })
    .await??;
    let provider = Arc::new(provider);

    let count = max_count.unwrap_or(requests.len());
    let bar = ProgressBar::new(count as u64);
    let mut success: usize = 0;
    let mut failure: usize = 0;
    for (i, request) in requests.into_iter().enumerate() {
        if let Some(max_count) = max_count {
            if i >= max_count {
                break;
            }
        }
        let p = provider.clone();
        let response = task::spawn_blocking(move || p.handle_request(request))
            .await?
            .map(|r| r.result);
        let response = jsonrpc::ResponseData::from(response);
        match response {
            jsonrpc::ResponseData::Success { .. } => success += 1,
            jsonrpc::ResponseData::Error { .. } => failure += 1,
        }
        if i % 100 == 0 {
            bar.inc(100);
        } else if i == count - 1 {
            bar.inc((count % 100) as u64);
        }
    }
    bar.finish();

    let elapsed = start.elapsed();

    println!(
        "Total time: {}s, Success: {}, Failure: {}",
        elapsed.as_secs(),
        success,
        failure
    );

    Ok(())
}

async fn load_requests(
    scenario_path: &Path,
) -> anyhow::Result<(ScenarioConfig, Vec<ProviderRequest<GenericChainSpec>>)> {
    println!("Loading requests from {scenario_path:?}");

    match load_gzipped_json(scenario_path.to_path_buf()).await {
        Ok(result) => Ok(result),
        Err(err) if err.to_string().contains("gzip") => load_json(scenario_path).await,
        err => err,
    }
}

async fn load_gzipped_json(
    scenario_path: PathBuf,
) -> anyhow::Result<(ScenarioConfig, Vec<ProviderRequest<GenericChainSpec>>)> {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    runtime::Handle::current()
        .spawn_blocking(move || {
            let reader = BufReader::new(File::open(scenario_path)?);
            let decoder = BufReader::new(GzDecoder::new(reader));

            let mut lines = decoder.lines();

            let first_line = lines
                .next()
                .context("Scenario file is empty")?
                .context("Invalid gzip")?;
            let config: ScenarioConfig = serde_json::from_str(&first_line)?;

            if let Some(chain_type) = &config.chain_type {
                anyhow::ensure!(
                    chain_type == GenericChainSpec::CHAIN_TYPE,
                    "Unsupported chain type: {chain_type}"
                );
            }

            let mut requests: Vec<ProviderRequest<GenericChainSpec>> = Vec::new();

            for gzipped_line in lines {
                let line = gzipped_line.context("Invalid gzip")?;
                let request: ProviderRequest<GenericChainSpec> = serde_json::from_str(&line)?;
                requests.push(request);
            }

            Ok((config, requests))
        })
        .await?
}

async fn load_json(
    scenario_path: &Path,
) -> anyhow::Result<(ScenarioConfig, Vec<ProviderRequest<GenericChainSpec>>)> {
    use tokio::io::AsyncBufReadExt;

    let reader = tokio::io::BufReader::new(tokio::fs::File::open(scenario_path).await?);
    let mut lines = reader.lines();

    let first_line = lines.next_line().await?.context("Scenario file is empty")?;
    let config: ScenarioConfig = serde_json::from_str(&first_line)?;

    if let Some(chain_type) = &config.chain_type {
        anyhow::ensure!(
            chain_type == GenericChainSpec::CHAIN_TYPE,
            "Unsupported chain type: {chain_type}"
        );
    }

    let mut requests: Vec<ProviderRequest<GenericChainSpec>> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        let request: ProviderRequest<GenericChainSpec> = serde_json::from_str(&line)?;
        requests.push(request);
    }

    Ok((config, requests))
}

#[derive_where(Clone, Default)]
struct DisabledLogger<ChainSpecT: RuntimeSpec> {
    _phantom: PhantomData<ChainSpecT>,
}

impl<ChainSpecT: RuntimeSpec> Logger<ChainSpecT> for DisabledLogger<ChainSpecT> {
    type BlockchainError = BlockchainErrorForChainSpec<GenericChainSpec>;

    fn is_enabled(&self) -> bool {
        false
    }

    fn set_is_enabled(&mut self, _is_enabled: bool) {}

    fn print_contract_decoding_error(
        &mut self,
        _error: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn print_method_logs(
        &mut self,
        _method: &str,
        _error: Option<&ProviderError<ChainSpecT>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}
