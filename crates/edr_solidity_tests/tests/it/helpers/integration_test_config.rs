use std::{path::PathBuf, str::FromStr, sync::Arc};

use foundry_compilers::{
    artifacts::{
        output_selection::{ContractOutputSelection, OutputSelection},
        BytecodeHash, DebuggingSettings, Libraries, ModelCheckerSettings, ModelCheckerTarget,
        Optimizer, OptimizerDetails, RevertStrings, Settings, SettingsMetadata, Severity,
    },
    cache::SOLIDITY_FILES_CACHE_FILENAME,
    compilers::{solc::SolcVersionManager, CompilerVersionManager},
    error::SolcError,
    remappings::{RelativeRemapping, Remapping},
    CompilerConfig, ConfigurableArtifacts, EvmVersion, Project, ProjectPathsConfig, Solc,
    SolcConfig,
};
use semver::Version;

use crate::helpers::solidity_error_code::SolidityErrorCode;

#[derive(Clone, Debug, PartialEq)]
pub struct IntegrationTestConfig {
    pub project_root: PathBuf,
    /// path of the source contracts dir, like `src` or `contracts`
    pub src: PathBuf,
    /// path of the test dir
    pub test: PathBuf,
    /// path of the script dir
    pub script: PathBuf,
    /// path to where artifacts shut be written to
    pub out: PathBuf,
    /// all library folders to include, `lib`, `node_modules`
    pub libs: Vec<PathBuf>,
    /// `Remappings` to use for this repo
    pub remappings: Vec<RelativeRemapping>,
    /// Whether to autodetect remappings by scanning the `libs` folders
    /// recursively
    pub auto_detect_remappings: bool,
    /// library addresses to link
    pub libraries: Vec<String>,
    /// whether to enable cache
    pub cache: bool,
    /// where the cache is stored if enabled
    pub cache_path: PathBuf,
    /// additional solc allow paths for `--allow-paths`
    pub allow_paths: Vec<PathBuf>,
    /// additional solc include paths for `--include-path`
    pub include_paths: Vec<PathBuf>,
    /// whether to force a `project.clean()`
    pub force: bool,
    /// evm version to use
    pub evm_version: EvmVersion,
    /// The Solc instance to use if any.
    ///
    /// This takes precedence over `auto_detect_solc`, if a version is set then
    /// this overrides auto-detection.
    ///
    /// **Note** for backwards compatibility reasons this also accepts
    /// `solc_version` from the toml file, see [`BackwardsCompatProvider`]
    pub solc: Option<SolcReq>,
    /// whether to autodetect the solc compiler version to use
    pub auto_detect_solc: bool,
    /// Offline mode, if set, network access (downloading solc) is disallowed.
    ///
    /// Relationship with `auto_detect_solc`:
    ///    - if `auto_detect_solc = true` and `offline = true`, the required
    ///      solc version(s) will be auto detected but if the solc version is
    ///      not installed, it will _not_ try to install it
    pub offline: bool,
    /// Whether to activate optimizer
    pub optimizer: bool,
    /// Sets the optimizer runs
    pub optimizer_runs: usize,
    /// Switch optimizer components on or off in detail.
    /// The "enabled" switch above provides two defaults which can be
    /// tweaked here. If "details" is given, "enabled" can be omitted.
    pub optimizer_details: Option<OptimizerDetails>,
    /// Model checker settings.
    pub model_checker: Option<ModelCheckerSettings>,
    /// list of solidity error codes to always silence in the compiler output
    pub ignored_error_codes: Vec<SolidityErrorCode>,
    /// list of file paths to ignore
    pub ignored_file_paths: Vec<PathBuf>,
    /// When true, compiler warnings are treated as errors
    pub deny_warnings: bool,
    /// Additional output selection for all contracts, such as "ir", "devdoc",
    /// "storageLayout", etc.
    ///
    /// See the [Solc Compiler Api](https://docs.soliditylang.org/en/latest/using-the-compiler.html#compiler-api) for more information.
    ///
    /// The following values are always set because they're required by
    /// `edr_solidity_tests`:
    /// ```json
    /// {
    ///   "*": [
    ///       "abi",
    ///       "evm.bytecode",
    ///       "evm.deployedBytecode",
    ///       "evm.methodIdentifiers"
    ///     ]
    /// }
    /// ```
    pub extra_output: Vec<ContractOutputSelection>,
    /// If set, a separate JSON file will be emitted for every contract
    /// depending on the selection, eg. `extra_output_files = ["metadata"]`
    /// will create a `metadata.json` for each contract in the project.
    ///
    /// See [Contract Metadata](https://docs.soliditylang.org/en/latest/metadata.html) for more information.
    ///
    /// The difference between `extra_output = ["metadata"]` and
    /// `extra_output_files = ["metadata"]` is that the former will include the
    /// contract's metadata in the contract's json artifact, whereas the latter
    /// will emit the output selection as separate files.
    pub extra_output_files: Vec<ContractOutputSelection>,
    /// Whether to print the names of the compiled contracts.
    pub names: bool,
    /// Whether to print the sizes of the compiled contracts.
    pub sizes: bool,
    /// If set to true, changes compilation pipeline to go through the Yul
    /// intermediate representation.
    pub via_ir: bool,
    /// Whether to include the AST as JSON in the compiler output.
    pub ast: bool,
    /// Whether to store the referenced sources in the metadata as literal data.
    pub use_literal_content: bool,
    /// Whether to include the metadata hash.
    ///
    /// The metadata hash is machine dependent. By default, this is set to [`BytecodeHash::None`] to allow for deterministic code, See: <https://docs.soliditylang.org/en/latest/metadata.html>
    pub bytecode_hash: BytecodeHash,
    /// Whether to append the metadata hash to the bytecode.
    ///
    /// If this is `false` and the `bytecode_hash` option above is not `None`
    /// solc will issue a warning.
    pub cbor_metadata: bool,
    /// How to treat revert (and require) reason strings.
    pub revert_strings: Option<RevertStrings>,
    /// Whether to compile in sparse mode
    ///
    /// If this option is enabled, only the required contracts/files will be
    /// selected to be included in solc's output selection, see also
    /// [`OutputSelection`](foundry_compilers::artifacts::output_selection::OutputSelection)
    pub sparse_mode: bool,
    /// Generates additional build info json files for every new build,
    /// containing the `CompilerInput` and `CompilerOutput`.
    pub build_info: bool,
    /// The path to the `build-info` directory that contains the build info json
    /// files.
    pub build_info_path: Option<PathBuf>,
}

impl IntegrationTestConfig {
    /// Serves as the entrypoint for obtaining the project.
    ///
    /// Returns the `Project` configured with all `solc` and path related
    /// values.
    ///
    /// *Note*: this also _cleans_ [`Project::cleanup`] the workspace if `force`
    /// is set to true.
    ///
    /// # Example
    ///
    /// ```
    /// use foundry_config::IntegrationTestConfig;
    /// let config = IntegrationTestConfig::with_root(".");
    /// let project = config.project();
    /// ```
    pub fn project(&self) -> Result<Project, SolcError> {
        self.create_project(self.cache, false)
    }

    /// Creates a [Project] with the given `cached` and `no_artifacts` flags
    fn create_project(&self, cached: bool, no_artifacts: bool) -> Result<Project, SolcError> {
        let project = Project::builder()
            .artifacts(self.configured_artifacts_handler())
            .paths(self.project_paths())
            .settings(
                SolcConfig::builder()
                    .settings(self.solc_settings()?)
                    .build()
                    .settings,
            )
            .ignore_error_codes(self.ignored_error_codes.iter().copied().map(Into::into))
            .ignore_paths(self.ignored_file_paths.clone())
            .set_compiler_severity_filter(if self.deny_warnings {
                Severity::Warning
            } else {
                Severity::Error
            })
            .set_offline(self.offline)
            .set_cached(cached && !self.build_info)
            .set_build_info(!no_artifacts && self.build_info)
            .set_no_artifacts(no_artifacts)
            .build(self.compiler_config()?)?;

        if self.force {
            project.cleanup()?;
        }

        Ok(project)
    }

    /// Ensures that the configured version is installed if explicitly set
    ///
    /// If `solc` is [`SolcReq::Version`] then this will download and install
    /// the solc version if it's missing, unless the `offline` flag is
    /// enabled, in which case an error is thrown.
    ///
    /// If `solc` is [`SolcReq::Local`] then this will ensure that the path
    /// exists.
    fn ensure_solc(&self) -> Result<Option<Solc>, SolcError> {
        if let Some(ref solc) = self.solc {
            let version_manager = SolcVersionManager::default();
            let solc = match solc {
                SolcReq::Version(version) => {
                    if let Ok(solc) = version_manager.get_installed(version) {
                        solc
                    } else {
                        if self.offline {
                            return Err(SolcError::msg(format!(
                                "can't install missing solc {version} in offline mode"
                            )));
                        }
                        version_manager.install(version)?
                    }
                }
                SolcReq::Local(solc) => {
                    if !solc.is_file() {
                        return Err(SolcError::msg(format!(
                            "`solc` {} does not exist",
                            solc.display()
                        )));
                    }
                    Solc::new(solc)?
                }
            };
            return Ok(Some(solc));
        }

        Ok(None)
    }

    fn project_paths(&self) -> ProjectPathsConfig {
        let mut builder = ProjectPathsConfig::builder()
            .cache(self.cache_path.join(SOLIDITY_FILES_CACHE_FILENAME))
            .sources(&self.src)
            .tests(&self.test)
            .scripts(&self.script)
            .artifacts(&self.out)
            .libs(self.libs.iter())
            .remappings(self.get_all_remappings())
            .allowed_path(&self.project_root)
            .allowed_paths(&self.libs)
            .allowed_paths(&self.allow_paths)
            .include_paths(&self.include_paths);

        if let Some(build_info_path) = &self.build_info_path {
            builder = builder.build_infos(build_info_path);
        }

        builder.build_with_root(&self.project_root)
    }

    /// Returns configuration for a compiler to use when setting up a [Project].
    fn compiler_config(&self) -> Result<CompilerConfig<Solc>, SolcError> {
        if let Some(solc) = self.ensure_solc()? {
            Ok(CompilerConfig::Specific(solc))
        } else {
            Ok(CompilerConfig::AutoDetect(Arc::new(
                SolcVersionManager::default(),
            )))
        }
    }

    /// Returns all configured [`Remappings`]
    ///
    /// **Note:** this will add an additional `<src>/=<src path>` remapping
    /// here, see [`Self::get_source_dir_remapping()`]
    ///
    /// So that
    ///
    /// ```solidity
    /// import "./math/math.sol";
    /// import "contracts/tokens/token.sol";
    /// ```
    ///
    /// in `contracts/contract.sol` are resolved to
    ///
    /// ```text
    /// contracts/tokens/token.sol
    /// contracts/math/math.sol
    /// ```
    fn get_all_remappings(&self) -> impl Iterator<Item = Remapping> + '_ {
        self.remappings.iter().map(|m| m.clone().into())
    }

    /// Returns the `Optimizer` based on the configured settings
    ///
    /// Note: optimizer details can be set independently of `enabled`
    /// See also: <https://github.com/foundry-rs/foundry/issues/7689>
    /// and  <https://github.com/ethereum/solidity/blob/bbb7f58be026fdc51b0b4694a6f25c22a1425586/docs/using-the-compiler.rst?plain=1#L293-L294>
    fn optimizer(&self) -> Optimizer {
        Optimizer {
            enabled: Some(self.optimizer),
            runs: Some(self.optimizer_runs),
            // we always set the details because `enabled` is effectively a specific details profile
            // that can still be modified
            details: self.optimizer_details.clone(),
        }
    }

    /// returns the [`foundry_compilers::ConfigurableArtifacts`] for this
    /// config, that includes the `extra_output` fields
    fn configured_artifacts_handler(&self) -> ConfigurableArtifacts {
        let mut extra_output = self.extra_output.clone();

        // Sourcify verification requires solc metadata output. Since, it doesn't
        // affect the UX & performance of the compiler, output the metadata files
        // by default.
        // For more info see: <https://github.com/foundry-rs/foundry/issues/2795>
        // Metadata is not emitted as separate file because this breaks typechain support: <https://github.com/foundry-rs/foundry/issues/2969>
        if !extra_output.contains(&ContractOutputSelection::Metadata) {
            extra_output.push(ContractOutputSelection::Metadata);
        }

        ConfigurableArtifacts::new(extra_output, self.extra_output_files.iter().cloned())
    }

    /// Parses all libraries in the form of
    /// `<file>:<lib>:<addr>`
    fn parsed_libraries(&self) -> Result<Libraries, SolcError> {
        Libraries::parse(&self.libraries)
    }

    /// Returns all libraries with applied remappings. Same as
    /// `self.solc_settings()?.libraries`.
    fn libraries_with_remappings(&self) -> Result<Libraries, SolcError> {
        Ok(self
            .parsed_libraries()?
            .with_applied_remappings(&self.project_paths()))
    }

    /// Returns the configured `solc` `Settings` that includes:
    /// - all libraries
    /// - the optimizer (including details, if configured)
    /// - evm version
    fn solc_settings(&self) -> Result<Settings, SolcError> {
        // By default if no targets are specifically selected the model checker uses all
        // targets. This might be too much here, so only enable assertion
        // checks. If users wish to enable all options they need to do so
        // explicitly.
        let mut model_checker = self.model_checker.clone();
        if let Some(model_checker_settings) = &mut model_checker {
            if model_checker_settings.targets.is_none() {
                model_checker_settings.targets = Some(vec![ModelCheckerTarget::Assert]);
            }
        }

        let mut settings = Settings {
            libraries: self.libraries_with_remappings()?,
            optimizer: self.optimizer(),
            evm_version: Some(self.evm_version),
            metadata: Some(SettingsMetadata {
                use_literal_content: Some(self.use_literal_content),
                bytecode_hash: Some(self.bytecode_hash),
                cbor_metadata: Some(self.cbor_metadata),
            }),
            debug: self.revert_strings.map(|revert_strings| DebuggingSettings {
                revert_strings: Some(revert_strings),
                // Not used.
                debug_info: Vec::new(),
            }),
            model_checker,
            via_ir: Some(self.via_ir),
            // Not used.
            stop_after: None,
            // Set in project paths.
            remappings: Vec::new(),
            // Set with `with_extra_output` below.
            output_selection: OutputSelection::default(),
        }
        .with_extra_output(self.configured_artifacts_handler().output_selection());

        // We're keeping AST in `--build-info` for backwards compatibility with HardHat.
        if self.ast || self.build_info {
            settings = settings.with_ast();
        }

        Ok(settings)
    }

    /// Creates a new Config that adds additional context extracted from the
    /// provided root.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        // autodetect paths
        let root = root.into();
        let paths = ProjectPathsConfig::builder().build_with_root::<()>(&root);
        let artifacts: PathBuf = paths.artifacts.file_name().unwrap().into();
        IntegrationTestConfig {
            project_root: paths.root,
            src: paths.sources.file_name().unwrap().into(),
            out: artifacts.clone(),
            libs: paths
                .libraries
                .into_iter()
                .map(|lib| lib.file_name().unwrap().into())
                .collect(),
            remappings: paths
                .remappings
                .into_iter()
                .map(|r| RelativeRemapping::new(r, &root))
                .collect(),
            test: "test".into(),
            script: "script".into(),
            cache: true,
            cache_path: "cache".into(),
            allow_paths: vec![],
            include_paths: vec![],
            force: false,
            evm_version: EvmVersion::Paris,
            solc: None,
            auto_detect_solc: true,
            offline: false,
            optimizer: true,
            optimizer_runs: 200,
            optimizer_details: None,
            model_checker: None,
            extra_output: Vec::default(),
            extra_output_files: Vec::default(),
            names: false,
            sizes: false,
            auto_detect_remappings: true,
            libraries: vec![],
            ignored_error_codes: vec![
                SolidityErrorCode::SpdxLicenseNotProvided,
                SolidityErrorCode::ContractExceeds24576Bytes,
                SolidityErrorCode::ContractInitCodeSizeExceeds49152Bytes,
                SolidityErrorCode::TransientStorageUsed,
            ],
            ignored_file_paths: vec![],
            deny_warnings: false,
            via_ir: false,
            ast: false,
            use_literal_content: false,
            bytecode_hash: BytecodeHash::Ipfs,
            cbor_metadata: true,
            revert_strings: None,
            sparse_mode: false,
            build_info: false,
            build_info_path: None,
        }
    }
}

/// Variants for selecting the [`Solc`] instance
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolcReq {
    /// Requires a specific solc version, that's either already installed (via
    /// `svm`) or will be auto installed (via `svm`)
    Version(Version),
    /// Path to an existing local solc installation
    Local(PathBuf),
}

impl<T: AsRef<str>> From<T> for SolcReq {
    fn from(s: T) -> Self {
        let s = s.as_ref();
        if let Ok(v) = Version::from_str(s) {
            SolcReq::Version(v)
        } else {
            SolcReq::Local(s.into())
        }
    }
}
