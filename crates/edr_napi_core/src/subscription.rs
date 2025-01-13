use derive_where::derive_where;
use edr_eth::B256;
use edr_provider::{time::CurrentTime, ProviderSpec, SubscriptionEvent};
use napi::{
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};

#[derive_where(Clone)]
pub struct Callback<ChainSpecT: ProviderSpec<CurrentTime> + 'static> {
    inner: ThreadsafeFunction<SubscriptionEvent<ChainSpecT>, ErrorStrategy::Fatal>,
}

impl<ChainSpecT: ProviderSpec<CurrentTime>> Callback<ChainSpecT> {
    pub fn new(env: &napi::Env, subscription_event_callback: JsFunction) -> napi::Result<Self> {
        let mut callback = subscription_event_callback.create_threadsafe_function(
            0,
            |ctx: ThreadSafeCallContext<SubscriptionEvent<ChainSpecT>>| {
                // SubscriptionEvent
                let mut event = ctx.env.create_object()?;

                ctx.env
                    .create_bigint_from_words(false, ctx.value.filter_id.as_limbs().to_vec())
                    .and_then(|filter_id| event.set_named_property("filterId", filter_id))?;

                let result = match ctx.value.result {
                    edr_provider::SubscriptionEventData::Logs(logs) => ctx.env.to_js_value(&logs),
                    edr_provider::SubscriptionEventData::NewHeads(block) => {
                        let block = ChainSpecT::RpcBlock::<B256>::from(block);
                        ctx.env.to_js_value(&block)
                    }
                    edr_provider::SubscriptionEventData::NewPendingTransactions(tx_hash) => {
                        ctx.env.to_js_value(&tx_hash)
                    }
                }?;

                event.set_named_property("result", result)?;

                Ok(vec![event])
            },
        )?;

        // Maintain a weak reference to the function to avoid the event loop from
        // exiting.
        callback.unref(env)?;

        Ok(Self { inner: callback })
    }

    pub fn call(&self, event: SubscriptionEvent<ChainSpecT>) {
        // This is blocking because it's important that the subscription events are
        // in-order
        self.inner.call(event, ThreadsafeFunctionCallMode::Blocking);
    }
}

/// Configuration for subscriptions.
pub struct Config {
    /// Callback to be called when a new event is received.
    pub subscription_callback: JsFunction,
}
