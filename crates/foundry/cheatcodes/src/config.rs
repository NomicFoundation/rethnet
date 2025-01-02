use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use alloy_primitives::Address;
use edr_common::fs::normalize_path;
use foundry_compilers::utils::canonicalize;
use foundry_evm_core::{contracts::ContractsByArtifact, opts::EvmOpts};
use semver::Version;

use super::{FsAccessKind, FsPermissions, Result, RpcEndpoints};
use crate::{cache::StorageCachingConfig, Vm::Rpc};

/// Additional, configurable context the `Cheatcodes` inspector has access to
///
/// This is essentially a subset of various `Config` settings `Cheatcodes` needs
/// to know.
#[derive(Clone, Debug)]
pub struct CheatsConfig {
    /// Whether the execution is in the context of a test run, gas snapshot or
    /// code coverage.
    pub execution_context: ExecutionContextConfig,
    /// Whether the FFI cheatcode is enabled.
    pub ffi: bool,
    /// Use the create 2 factory in all cases including tests and
    /// non-broadcasting scripts.
    pub always_use_create_2_factory: bool,
    /// Sets a timeout for vm.prompt cheatcodes
    pub prompt_timeout: Duration,
    /// Optional RPC cache path. If this is none, then no RPC calls will be
    /// cached, otherwise data is cached to `<rpc_cache_path>/<chain
    /// id>/<block number>`. Caching can be disabled for specific chains
    /// with `rpc_storage_caching`.
    pub rpc_cache_path: Option<PathBuf>,
    /// RPC storage caching settings determines what chains and endpoints to
    /// cache
    pub rpc_storage_caching: StorageCachingConfig,
    /// All known endpoints and their aliases
    pub rpc_endpoints: RpcEndpoints,
    /// Filesystem permissions for cheatcodes like `writeFile`, `readFile`
    pub fs_permissions: FsPermissions,
    /// Project root
    pub project_root: PathBuf,
    /// How the evm was configured by the user
    pub evm_opts: EvmOpts,
    /// Address labels from config
    pub labels: HashMap<Address, String>,
    /// Solidity compilation artifacts.
    pub available_artifacts: Arc<ContractsByArtifact>,
    /// Version of the script/test contract which is currently running.
    pub running_version: Option<Version>,
}

/// Solidity test execution contexts.
#[derive(Clone, Debug, Default)]
pub enum ExecutionContextConfig {
    /// Test execution context.
    Test,
    /// Code coverage execution context.
    Coverage,
    /// Gas snapshot execution context.
    Snapshot,
    /// Unknown execution context.
    #[default]
    Unknown,
}

/// Configuration options specific to cheat codes.
#[derive(Clone, Debug, Default)]
pub struct CheatsConfigOptions {
    /// Solidity test execution contexts.
    pub execution_context: ExecutionContextConfig,
    /// Multiple rpc endpoints and their aliases
    pub rpc_endpoints: RpcEndpoints,
    /// Optional RPC cache path. If this is none, then no RPC calls will be
    /// cached, otherwise data is cached to `<rpc_cache_path>/<chain
    /// id>/<block number>`. Caching can be disabled for specific chains
    /// with `rpc_storage_caching`.
    pub rpc_cache_path: Option<PathBuf>,
    /// RPC storage caching settings determines what chains and endpoints to
    /// cache
    pub rpc_storage_caching: StorageCachingConfig,
    /// Configures the permissions of cheat codes that touch the file system.
    ///
    /// This includes what operations can be executed (read, write)
    pub fs_permissions: FsPermissions,
    /// Sets a timeout in seconds for vm.prompt cheatcodes
    pub prompt_timeout: u64,
    /// Address labels
    pub labels: HashMap<Address, String>,
}

impl CheatsConfig {
    /// Extracts the necessary settings from the Config
    pub fn new(
        project_root: PathBuf,
        config: CheatsConfigOptions,
        evm_opts: EvmOpts,
        available_artifacts: Arc<ContractsByArtifact>,
        running_version: Option<Version>,
    ) -> Self {
        let CheatsConfigOptions {
            execution_context,
            rpc_endpoints,
            rpc_cache_path,
            prompt_timeout,
            rpc_storage_caching,
            fs_permissions,
            labels,
        } = config;

        let fs_permissions = fs_permissions.joined(&project_root);

        Self {
            execution_context,
            ffi: evm_opts.ffi,
            always_use_create_2_factory: evm_opts.always_use_create_2_factory,
            prompt_timeout: Duration::from_secs(prompt_timeout),
            rpc_cache_path,
            rpc_storage_caching,
            rpc_endpoints,
            fs_permissions,
            project_root,
            evm_opts,
            labels,
            available_artifacts,
            running_version,
        }
    }

    /// Attempts to canonicalize (see [`std::fs::canonicalize`]) the path.
    ///
    /// Canonicalization fails for non-existing paths, in which case we just
    /// normalize the path.
    pub fn normalized_path(&self, path: impl AsRef<Path>) -> PathBuf {
        let path = self.project_root.join(path);
        canonicalize(&path).unwrap_or_else(|_err| normalize_path(&path))
    }

    /// Returns true if the given path is allowed, if any path `allowed_paths`
    /// is an ancestor of the path
    ///
    /// We only allow paths that are inside  allowed paths. To prevent path
    /// traversal ("../../etc/passwd") we canonicalize/normalize the path
    /// first. We always join with the configured root directory.
    pub fn is_path_allowed(&self, path: impl AsRef<Path>, kind: FsAccessKind) -> bool {
        self.is_normalized_path_allowed(&self.normalized_path(path), kind)
    }

    fn is_normalized_path_allowed(&self, path: &Path, kind: FsAccessKind) -> bool {
        self.fs_permissions.is_path_allowed(path, kind)
    }

    /// Returns an error if no access is granted to access `path`, See also
    /// [`Self::is_path_allowed`]
    ///
    /// Returns the normalized version of `path`, see
    /// [`CheatsConfig::normalized_path`]
    pub fn ensure_path_allowed(
        &self,
        path: impl AsRef<Path>,
        kind: FsAccessKind,
    ) -> Result<PathBuf> {
        let path = path.as_ref();
        let normalized = self.normalized_path(path);
        ensure!(
            self.is_normalized_path_allowed(&normalized, kind),
            "the path {} is not allowed to be accessed for {kind} operations",
            normalized
                .strip_prefix(&self.project_root)
                .unwrap_or(path)
                .display()
        );
        Ok(normalized)
    }

    /// Returns true if the given `path` is the project's foundry.toml file
    ///
    /// Note: this should be called with normalized path
    pub fn is_foundry_toml(&self, path: impl AsRef<Path>) -> bool {
        const FILE_NAME: &str = "foundry.toml";

        // path methods that do not access the filesystem are such as
        // [`Path::starts_with`], are case-sensitive no matter the platform or
        // filesystem. to make this case-sensitive we convert the underlying
        // `OssStr` to lowercase checking that `path` and `foundry.toml` are the
        // same file by comparing the FD, because it may not exist
        let foundry_toml = self.project_root.join(FILE_NAME);
        Path::new(&foundry_toml.to_string_lossy().to_lowercase())
            .starts_with(Path::new(&path.as_ref().to_string_lossy().to_lowercase()))
    }

    /// Same as [`Self::is_foundry_toml`] but returns an `Err` if
    /// [`Self::is_foundry_toml`] returns true
    pub fn ensure_not_foundry_toml(&self, path: impl AsRef<Path>) -> Result<()> {
        ensure!(
            !self.is_foundry_toml(path),
            "access to `foundry.toml` is not allowed"
        );
        Ok(())
    }

    /// Returns the RPC to use
    ///
    /// If `url_or_alias` is a known alias in the `RpcEndpoints` then it
    /// returns the corresponding URL of that alias. otherwise this assumes
    /// `url_or_alias` is itself a URL if it starts with a `http` or `ws`
    /// scheme.
    ///
    /// If the url is a path to an existing file, it is also considered a valid
    /// RPC URL, IPC path.
    ///
    /// # Errors
    ///
    ///  - Returns an error if `url_or_alias` is a known alias but references an
    ///    unresolved env var.
    ///  - Returns an error if `url_or_alias` is not an alias but does not start
    ///    with a `http` or `ws` `scheme` and is not a path to an existing file
    pub fn rpc_url(&self, url_or_alias: &str) -> Result<String> {
        match self.rpc_endpoints.get(url_or_alias) {
            Some(endpoint_config) => {
                if let Some(url) = endpoint_config.endpoint.as_url() {
                    Ok(url.into())
                } else {
                    Err(fmt_err!("unresolved env var in rpc url: {url_or_alias}"))
                }
            }
            None => {
                // check if it's a URL or a path to an existing file to an ipc socket
                if url_or_alias.starts_with("http") ||
                    url_or_alias.starts_with("ws") ||
                    // check for existing ipc file
                    Path::new(url_or_alias).exists()
                {
                    Ok(url_or_alias.into())
                } else {
                    Err(fmt_err!("invalid rpc url: {url_or_alias}"))
                }
            }
        }
    }

    /// Returns all the RPC urls and their alias.
    pub fn rpc_urls(&self) -> Result<Vec<Rpc>> {
        let mut urls = Vec::with_capacity(self.rpc_endpoints.len());
        for alias in self.rpc_endpoints.keys() {
            let url = self.rpc_url(alias)?;
            urls.push(Rpc {
                key: alias.clone(),
                url,
            });
        }
        Ok(urls)
    }
}

impl Default for CheatsConfig {
    fn default() -> Self {
        Self {
            execution_context: ExecutionContextConfig::default(),
            ffi: false,
            always_use_create_2_factory: false,
            prompt_timeout: Duration::from_secs(120),
            rpc_cache_path: None,
            rpc_storage_caching: StorageCachingConfig::default(),
            rpc_endpoints: RpcEndpoints::default(),
            fs_permissions: FsPermissions::default(),
            project_root: PathBuf::default(),
            evm_opts: EvmOpts::default(),
            labels: HashMap::default(),
            available_artifacts: Arc::<ContractsByArtifact>::default(),
            running_version: Option::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cache::StorageCachingConfig, PathPermission};

    fn config(root: &str, fs_permissions: FsPermissions) -> CheatsConfig {
        let cheats_config_options = CheatsConfigOptions {
            execution_context: ExecutionContextConfig::default(),
            rpc_endpoints: RpcEndpoints::default(),
            rpc_cache_path: None,
            rpc_storage_caching: StorageCachingConfig::default(),
            fs_permissions,
            prompt_timeout: 0,
            labels: HashMap::default(),
        };

        CheatsConfig::new(
            PathBuf::from(root),
            cheats_config_options,
            EvmOpts::default(),
            Arc::<ContractsByArtifact>::default(),
            None,
        )
    }

    #[test]
    fn test_allowed_paths() {
        fn test_cases(config: CheatsConfig) {
            assert!(config
                .ensure_path_allowed("./t.txt", FsAccessKind::Read)
                .is_ok());
            assert!(config
                .ensure_path_allowed("./t.txt", FsAccessKind::Write)
                .is_ok());
            assert!(config
                .ensure_path_allowed("../root/t.txt", FsAccessKind::Read)
                .is_ok());
            assert!(config
                .ensure_path_allowed("../root/t.txt", FsAccessKind::Write)
                .is_ok());
            assert!(config
                .ensure_path_allowed("../../root/t.txt", FsAccessKind::Read)
                .is_err());
            assert!(config
                .ensure_path_allowed("../../root/t.txt", FsAccessKind::Write)
                .is_err());

            assert!(config
                .ensure_path_allowed("/my/project/root/t.txt", FsAccessKind::Read)
                .is_ok());
            assert!(config
                .ensure_path_allowed("/my/project/root/../root/t.txt", FsAccessKind::Write)
                .is_ok());

            assert!(config
                .ensure_path_allowed("/other/project/root/t.txt", FsAccessKind::Read)
                .is_err());
        }

        let root = "/my/project/root/";

        test_cases(config(
            root,
            FsPermissions::new(vec![PathPermission::read_write("./")]),
        ));

        test_cases(config(
            root,
            FsPermissions::new(vec![PathPermission::read_write("/my/project/root")]),
        ));
    }

    #[test]
    fn test_is_foundry_toml() {
        let root = "/my/project/root/";
        let config = config(
            root,
            FsPermissions::new(vec![PathPermission::read_write("./")]),
        );

        let f = format!("{root}foundry.toml");
        assert!(config.is_foundry_toml(f));

        let f = format!("{root}Foundry.toml");
        assert!(config.is_foundry_toml(f));

        let f = format!("{root}lib/other/foundry.toml");
        assert!(!config.is_foundry_toml(f));
    }
}
