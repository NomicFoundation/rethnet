use std::future::Future;

use tokio::runtime;

#[repr(transparent)]
pub struct RuntimeHandle(runtime::Handle);

impl RuntimeHandle {
    pub fn create_scope<T: Send + 'static>(&'_ self) -> async_scoped::Scope<'_, T, &'_ Self> {
        unsafe { async_scoped::Scope::create(self) }
    }
}

impl From<runtime::Handle> for RuntimeHandle {
    fn from(value: runtime::Handle) -> Self {
        Self(value)
    }
}

unsafe impl<'runtime, T: Send + 'static> async_scoped::spawner::Spawner<T>
    for &'runtime RuntimeHandle
{
    type FutureOutput = Result<T, tokio::task::JoinError>;
    type SpawnHandle = tokio::task::JoinHandle<T>;

    fn spawn<F: Future<Output = T> + Send + 'static>(&self, f: F) -> Self::SpawnHandle {
        self.0.spawn(f)
    }
}

unsafe impl<'runtime, T: Send + 'static> async_scoped::spawner::FuncSpawner<T>
    for &'runtime RuntimeHandle
{
    type FutureOutput = Result<T, tokio::task::JoinError>;
    type SpawnHandle = tokio::task::JoinHandle<T>;

    fn spawn_func<F: FnOnce() -> T + Send + 'static>(&self, f: F) -> Self::SpawnHandle {
        self.0.spawn_blocking(f)
    }
}

unsafe impl<'runtime> async_scoped::spawner::Blocker for &'runtime RuntimeHandle {
    fn block_on<T, F: Future<Output = T>>(&self, f: F) -> T {
        self.0.block_on(f)
    }
}
