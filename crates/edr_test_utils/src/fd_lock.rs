//! File locking utilities.

use std::{
    fs::{File, OpenOptions},
    path::Path,
};

pub use fd_lock::RwLock;

/// Creates a new lock file at the given path.
pub fn new_fd_lock(lock_path: impl AsRef<Path>) -> RwLock<File> {
    fn new_lock(lock_path: &Path) -> RwLock<File> {
        let lock_file = pretty_err(
            lock_path,
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(lock_path),
        );
        RwLock::new(lock_file)
    }
    new_lock(lock_path.as_ref())
}

#[track_caller]
fn pretty_err<T, E: std::error::Error>(path: impl AsRef<Path>, res: Result<T, E>) -> T {
    match res {
        Ok(t) => t,
        Err(err) => panic!("{}: {err}", path.as_ref().display()),
    }
}
