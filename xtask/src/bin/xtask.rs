use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Error};
use once_cell::sync::Lazy;
use structopt::StructOpt;
use tracing_subscriber::EnvFilter;
use xtask::{runtime::Runtime, CompilationMode};

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("cranelift_codegen=warn".parse()?)
                .add_directive("wasmtime_cranelift=warn".parse()?)
                .add_directive("regalloc=warn".parse()?),
        )
        .without_time()
        .init();

    let cmd = Command::from_args();

    tracing::debug!(?cmd, "Starting");

    match cmd {
        Command::Dist(d) => d.execute(),
        Command::Metadata(m) => m.execute(),
        Command::Doc(d) => d.execute(),
        Command::Graph(g) => g.execute(),
    }
}

#[derive(Debug, StructOpt)]
enum Command {
    /// Compile all proc-blocks to WebAssembly and generate a manifest file.
    Dist(Dist),
    /// Extract the metadata from a proc-block.
    Metadata(Metadata),
    /// Generate API documentation for one or more proc-blocks.
    Doc(Doc),
    /// 
    Graph(Graph),
}

#[derive(Debug, StructOpt)]
struct Dist {
    /// The top-level `Cargo.toml` file.
    #[structopt(long, default_value = "./Cargo.toml")]
    workspace_root: PathBuf,
    /// Compile the proc-blocks without performing any runtime or code size
    /// optimisations.
    #[structopt(long)]
    debug: bool,
    /// Where to write compiled proc-blocks to.
    #[structopt(short, long, default_value = &*DIST_DIR)]
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
struct Metadata {
    /// The WebAssembly module to load.
    #[structopt(parse(from_os_str))]
    proc_block: PathBuf,
}

impl Metadata {
    fn execute(self) -> Result<(), Error> {
        let wasm = std::fs::read(&self.proc_block).with_context(|| {
            format!("Unable to read \"{}\"", self.proc_block.display())
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

#[derive(Debug, StructOpt)]
struct Graph {
    /// The WebAssembly module to load.
    #[structopt(parse(from_os_str))]
    rune: PathBuf,
    #[structopt(parse(try_from_str))]
    args: Vec<Argument>,
}

impl Graph {
    fn execute(self) -> Result<(), Error> {
        let wasm = std::fs::read(&self.rune).with_context(|| {
            format!("Unable to read \"{}\"", self.rune.display())
        })?;

        let mut runtime = Runtime::load(&wasm)
            .context("Unable to load the WebAssembly module")?;

        let arguments: HashMap<_, _> =
            self.args.into_iter().map(|a| (a.key, a.value)).collect();

        let info = runtime
            .graph(arguments)
            .context("Unable to infer the input and output tensors")?;

        let json = serde_json::to_string_pretty(&info)
            .context("Unable to serialize the metadata to JSON")?;

        println!("{}", json);

        Ok(())
    }
}

#[derive(Debug)]
struct Argument {
    key: String,
    value: String,
}

impl FromStr for Argument {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (key, value) =
            s.split_once('=').context("Expected a `key=value` pair")?;

        Ok(Argument {
            key: key.to_string(),
            value: value.to_string(),
        })
    }
}

#[derive(Debug, StructOpt)]
struct Doc {
    /// Where to write the generated documentation.
    #[structopt(short, long, default_value = &*DOCS_DIR)]
    out_dir: PathBuf,
    /// The WebAssembly modules to document.
    #[structopt(parse(from_os_str))]
    proc_blocks: Vec<PathBuf>,
}

impl Doc {
    fn execute(self) -> Result<(), Error> {
        std::fs::create_dir_all(&self.out_dir).with_context(|| {
            format!(
                "Unable to create the \"{}\" directory",
                self.out_dir.display()
            )
        })?;

        for proc_block in &self.proc_blocks {
            let _span = tracing::info_span!("documenting").entered();

            let filename = proc_block
                .file_stem()
                .and_then(|s| s.to_str())
                .context("Unable to get the filename")?;

            tracing::info!(
                name = filename,
                path = %proc_block.display(),
                "Generating Documentation",
            );

            let wasm = std::fs::read(proc_block).with_context(|| {
                format!("Unable to read \"{}\"", proc_block.display())
            })?;

            tracing::debug!(
                bytes_read = wasm.len(),
                "Read the module into memory"
            );

            let mut r = Runtime::load(&wasm)
                .context("Unable to load the proc-block")?;
            let meta = r
                .metadata()
                .context("Unable to extract the proc-block metadata")?;

            let dest = self.out_dir.join(filename).with_extension("md");
            tracing::debug!(path = %dest.display(), "Opened file for writing");

            let f = File::create(&dest).with_context(|| {
                format!("Unable to open \"{}\" for writing", dest.display())
            })?;
            let mut writer = BufWriter::new(f);

            xtask::document(&mut writer, &meta)
                .context("Unable to generate the documentation")?;

            writer.flush().context("Flush failed")?;
        }

        Ok(())
    }
}

static PROJECT_ROOT: Lazy<String> = Lazy::new(|| {
    for ancestor in Path::new(env!("CARGO_MANIFEST_DIR")).ancestors() {
        if ancestor.join(".git").exists() {
            return ancestor.display().to_string();
        }
    }

    String::from(".")
});

static TARGET_DIR: Lazy<String> = Lazy::new(|| {
    Path::new(PROJECT_ROOT.as_str())
        .join("target")
        .display()
        .to_string()
});

static DIST_DIR: Lazy<String> = Lazy::new(|| {
    Path::new(TARGET_DIR.as_str())
        .join("proc-blocks")
        .display()
        .to_string()
});

static DOCS_DIR: Lazy<String> = Lazy::new(|| {
    Path::new(TARGET_DIR.as_str())
        .join("proc-block-docs")
        .display()
        .to_string()
});
