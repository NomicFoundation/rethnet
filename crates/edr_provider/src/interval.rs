use std::{marker::PhantomData, sync::Arc};

use edr_eth::{result::InvalidTransaction, transaction::TransactionValidation};
use edr_evm::spec::RuntimeSpec;
use tokio::{
    runtime,
    sync::{oneshot, Mutex},
    task::JoinHandle,
    time::Instant,
};

use crate::{
    data::ProviderData, spec::SyncProviderSpec, time::TimeSinceEpoch, IntervalConfig, ProviderError,
};

/// Type for interval mining on a separate thread.
pub struct IntervalMiner<ChainSpecT: RuntimeSpec, TimerT> {
    inner: Option<Inner<ChainSpecT>>,
    runtime: runtime::Handle,
    phantom: PhantomData<TimerT>,
}

/// Inner type for interval mining on a separate thread, required for
/// implementation of `Drop`.
struct Inner<ChainSpecT: RuntimeSpec> {
    cancellation_sender: oneshot::Sender<()>,
    background_task: JoinHandle<Result<(), ProviderError<ChainSpecT>>>,
}

impl<
        ChainSpecT: SyncProviderSpec<
            TimerT,
            BlockEnv: Default,
            SignedTransaction: Default
                                   + TransactionValidation<
                ValidationError: From<InvalidTransaction> + PartialEq,
            >,
        >,
        TimerT: Clone + TimeSinceEpoch,
    > IntervalMiner<ChainSpecT, TimerT>
{
    pub fn new(
        runtime: runtime::Handle,
        config: IntervalConfig,
        data: Arc<Mutex<ProviderData<ChainSpecT, TimerT>>>,
    ) -> Self {
        let (cancellation_sender, cancellation_receiver) = oneshot::channel();
        let background_task = runtime
            .spawn(async move { interval_mining_loop(config, data, cancellation_receiver).await });

        Self {
            inner: Some(Inner {
                cancellation_sender,
                background_task,
            }),
            runtime,
            phantom: PhantomData,
        }
    }
}

#[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
async fn interval_mining_loop<
    ChainSpecT: SyncProviderSpec<
        TimerT,
        BlockEnv: Default,
        SignedTransaction: Default
                               + TransactionValidation<
            ValidationError: From<InvalidTransaction> + PartialEq,
        >,
    >,
    TimerT: Clone + TimeSinceEpoch,
>(
    config: IntervalConfig,
    data: Arc<Mutex<ProviderData<ChainSpecT, TimerT>>>,
    mut cancellation_receiver: oneshot::Receiver<()>,
) -> Result<(), ProviderError<ChainSpecT>> {
    let mut now = Instant::now();
    loop {
        let delay = config.generate_interval();
        let deadline = now + std::time::Duration::from_millis(delay);

        tokio::select! {
            _ = &mut cancellation_receiver => return Ok(()),
            _ = tokio::time::sleep_until(deadline) => {
                tokio::select! {
                    // Check whether the interval miner needs to be destroyed
                    _ = &mut cancellation_receiver => return Ok(()),
                    mut data = data.lock() => {
                        now = Instant::now();

                        if let Err(error) = data.interval_mine() {
                            log::error!("Unexpected error while performing interval mining: {error}");
                            return Err(error);
                        }

                        Result::<(), ProviderError<ChainSpecT>>::Ok(())
                    }
                }
            },
        }?;
    }
}

impl<ChainSpecT: RuntimeSpec, TimerT> Drop for IntervalMiner<ChainSpecT, TimerT> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    fn drop(&mut self) {
        if let Some(Inner {
            cancellation_sender,
            background_task: task,
        }) = self.inner.take()
        {
            cancellation_sender
                .send(())
                .expect("Failed to send cancellation signal");

            let _result = tokio::task::block_in_place(move || self.runtime.block_on(task))
                .expect("Failed to join interval mininig task");
        }
    }
}
