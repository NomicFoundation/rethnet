/// Initializes tracing for Solidity tests.
pub fn init_tracing_for_solidity_tests() {
    let _ = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}
