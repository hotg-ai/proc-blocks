use crate::{
    build::CompiledModule, proc_block_v2::Metadata, runtime::ProcBlockModule,
};
use anyhow::{Context, Error};
use serde::Serialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{Seek, SeekFrom},
    path::Path,
};

pub fn generate_manifest(
    modules: Vec<CompiledModule>,
) -> Result<Manifest, Error> {
    let mut manifest = Manifest::default();

    for module in modules {
        let CompiledModule { name, mut module } = module;
        let _span = tracing::info_span!("Extracting metadata", module = %name)
            .entered();

        let serialized = module.emit_wasm();
        let metadata = extract_metadata(&serialized).with_context(|| {
            format!("Unable to extract metadata from \"{}\"", name)
        })?;
        tracing::debug!(
            %metadata.name,
            %metadata.version,
            "Extracted metadata for proc-block",
        );

        let filename = format!("{}.wasm", name);
        manifest.serialized.insert(filename.clone(), serialized);
        manifest.metadata.insert(filename, metadata);
    }

    Ok(manifest)
}

fn extract_metadata(serialized: &[u8]) -> Result<Metadata, Error> {
    ProcBlockModule::load(serialized)?.metadata()
}

#[derive(Default)]
pub struct Manifest {
    metadata: HashMap<String, Metadata>,
    serialized: HashMap<String, Vec<u8>>,
}

impl Manifest {
    #[tracing::instrument(skip(self))]
    pub fn write_to_disk(&self, dir: &Path) -> Result<(), Error> {
        std::fs::create_dir_all(dir).with_context(|| {
            format!("Unable to create the \"{}\" directory", dir.display())
        })?;

        for (name, wasm) in &self.serialized {
            let filename = dir.join(&name);
            std::fs::write(&filename, wasm).with_context(|| {
                format!("Unable to save to \"{}\"", filename.display())
            })?;
        }

        let names: Vec<_> = self.metadata.keys().collect();
        save_json(dir.join("manifest.json"), &names)
            .context("Unable to save the manifest")?;

        Ok(())
    }
}

fn save_json(
    path: impl AsRef<Path>,
    value: &impl Serialize,
) -> Result<(), Error> {
    let path = path.as_ref();

    let mut f = File::create(path).with_context(|| {
        format!("Unable to open \"{}\" for writing", path.display())
    })?;

    serde_json::to_writer_pretty(&mut f, &value)?;

    let len = f.seek(SeekFrom::End(0))?;
    tracing::debug!(bytes_written = len, path = %path.display(), "Saved");

    Ok(())
}
