use std::path::PathBuf;

use anyhow::{Context, Error};
use structopt::StructOpt;
use tracing_subscriber::EnvFilter;
use xtask::{runtime::Runtime, CompilationMode};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .without_time()
        .init();

    let cmd = Command::from_args();

    tracing::debug!(?cmd, "Starting");

    match cmd {
        Command::Dist(d) => d.execute(),
        Command::Metadata(m) => m.execute(),
    }
}

#[derive(Debug, StructOpt)]
pub enum Command {
    /// Compile all proc-blocks to WebAssembly and generate a manifest file.
    Dist(Dist),
    Metadata(Metadata),
}

#[derive(Debug, StructOpt)]
pub struct Dist {
    /// The top-level `Cargo.toml` file.
    #[structopt(long, default_value = "./Cargo.toml")]
    workspace_root: PathBuf,
    /// Compile the proc-blocks without performing any runtime or code size
    /// optimisations.
    #[structopt(long)]
    debug: bool,
    /// Where to write compiled proc-blocks to.
    #[structopt(short, long, default_value = ".")]
    out_dir: PathBuf,
}

impl Dist {
    fn execute(self) -> Result<(), Error> {
        let proc_blocks =
            xtask::discover_proc_block_manifests(&self.workspace_root)
                .context("Unable to find proc-blocks")?;

        let mode = if self.debug {
            CompilationMode::Debug
        } else {
            CompilationMode::Release
        };

        let mut wasm_modules = proc_blocks.compile(mode)?;

        if !self.debug {
            tracing::info!("Stripping custom sections to reduce binary size");
            wasm_modules.iter_mut().for_each(|m| m.strip());
        }

        if let Some(parent) = self.out_dir.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Unable to create the \"{}\" directory",
                    parent.display()
                )
            })?;
        }

        tracing::info!("Creating the release bundle");
        let bundle = xtask::generate_manifest(wasm_modules)?;

        bundle
            .write_to_disk(&self.out_dir)
            .context("Unable to write the metadata to disk")?;

        Ok(())
    }
}

#[derive(Debug, StructOpt)]
pub struct Metadata {
    /// The WebAssembly module to load.
    #[structopt(parse(from_os_str))]
    rune: PathBuf,
}

impl Metadata {
    fn execute(self) -> Result<(), Error> {
        let wasm = std::fs::read(&self.rune).with_context(|| {
            format!("Unable to read \"{}\"", self.rune.display())
        })?;

        let mut runtime = Runtime::load(&wasm)
            .context("Unable to load the WebAssembly module")?;

        let metadata = runtime
            .metadata()
            .context("Unable to determine the metadata")?;

        let json = serde_json::to_string_pretty(&metadata)
            .context("Unable to serialize the metadata to JSON")?;

        println!("{}", json);

        Ok(())
    }
}
