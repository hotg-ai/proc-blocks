use anyhow::{Context, Error};
use cargo_metadata::{CargoOpt, Metadata, MetadataCommand, Package};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
use walrus::{Module, ModuleCustomSections};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompilationMode {
    Debug,
    Release,
}

impl CompilationMode {
    pub(crate) fn dir(&self) -> &str {
        match self {
            CompilationMode::Debug => "debug",
            CompilationMode::Release => "release",
        }
    }
}

pub fn discover_proc_block_manifests(
    workspace_root: &Path,
) -> Result<ProcBlocks, Error> {
    let _span = tracing::info_span!("Discover").entered();

    let metadata = manifest(workspace_root)?;
    let mut packages = Vec::new();

    for pkg_id in &metadata.workspace_members {
        let package = &metadata[pkg_id];

        let is_proc_block = package
            .dependencies
            .iter()
            .any(|dep| dep.name == "hotg-rune-proc-blocks");

        if is_proc_block {
            tracing::debug!(name = %package.name, "Found a proc-block");
            packages.push(package.clone());
        }
    }

    Ok(ProcBlocks {
        packages,
        target_dir: metadata.target_directory.into_std_path_buf(),
        workspace_root: workspace_root.to_path_buf(),
    })
}

#[derive(Debug)]
pub struct ProcBlocks {
    workspace_root: PathBuf,
    packages: Vec<Package>,
    target_dir: PathBuf,
}

impl ProcBlocks {
    /// Compile all the proc-blocks to WebAssembly and parse them as
    /// [`walrus::Module`]s.
    pub fn compile(
        &self,
        mode: CompilationMode,
    ) -> Result<Vec<CompiledModule>, Error> {
        let _span = tracing::info_span!("Compile").entered();
        tracing::info!("Compiling proc-blocks to WebAssembly");

        let cargo =
            std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

        let mut libs = Vec::new();

        for package in &self.packages {
            let mut cmd = Command::new(&cargo);
            cmd.arg("rustc")
                .arg("--manifest-path")
                .arg(&package.manifest_path)
                .arg("--lib")
                .arg("--target=wasm32-unknown-unknown")
                .arg("--features=metadata")
                .arg("-Zunstable-options")
                .arg("--crate-type=cdylib");

            match mode {
                CompilationMode::Release => {
                    cmd.arg("--release");
                },
                CompilationMode::Debug => {},
            }

            tracing::debug!(command = ?cmd, "Running cargo build");

            let status = cmd.status().with_context(|| {
                format!(
                    "Unable to start \"{}\"",
                    cmd.get_program().to_string_lossy()
                )
            })?;

            tracing::debug!(exit_code = ?status.code(), "Cargo build completed");

            if !status.success() {
                anyhow::bail!("Compilation failed");
            }

            libs.push(&package.name);
        }

        tracing::debug!(?libs);

        let artifact_dir = self
            .target_dir
            .join("wasm32-unknown-unknown")
            .join(mode.dir());

        let mut modules = Vec::new();

        for lib in libs {
            let filename = artifact_dir
                .join(lib.replace("-", "_"))
                .with_extension("wasm");
            tracing::debug!(
                filename = %filename.display(),
                "Loading WebAssembly module",
            );

            let module = Module::from_file(&filename).with_context(|| {
                format!("Unable to parse \"{}\"", filename.display())
            })?;
            modules.push(CompiledModule {
                name: lib.clone(),
                module,
            });
        }

        Ok(modules)
    }
}

pub struct CompiledModule {
    pub name: String,
    pub module: Module,
}

impl CompiledModule {
    pub fn strip(&mut self) {
        let _span = tracing::info_span!("Strip").entered();

        remove_custom_sections(&mut self.module.customs);
    }

    pub fn serialize(self) -> (String, Vec<u8>) {
        let CompiledModule { name, mut module } = self;
        (name, module.emit_wasm())
    }
}

fn manifest(manifest_path: &Path) -> Result<Metadata, Error> {
    let mut cmd = MetadataCommand::new();

    cmd.manifest_path(manifest_path)
        .features(CargoOpt::SomeFeatures(vec!["metadata".to_string()]));

    tracing::debug!(
        manifest = %manifest_path.display(),
        command = ?cmd.cargo_command(),
        "Inspecting the manifest",
    );

    let metadata = cmd.exec().with_context(|| {
        format!(
            "Unable to determine the cargo metadata for \"{}\"",
            manifest_path.display()
        )
    })?;

    Ok(metadata)
}

fn remove_custom_sections(customs: &mut ModuleCustomSections) {
    let to_remove: Vec<_> = customs
        .iter()
        .filter_map(|(_, section)| {
            if section.name().starts_with(".debug_") {
                Some(section.name().to_string())
            } else {
                None
            }
        })
        .collect();

    for name in to_remove {
        tracing::debug!(%name, "Removing custom section",);
        customs.remove_raw(&name);
    }
}
