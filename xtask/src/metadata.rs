use std::path::Path;

use anyhow::{Context, Error};

use crate::CompiledModule;

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
        manifest.0.push((metadata, serialized));
    }

    Ok(manifest)
}

fn extract_metadata(serialized: &[u8]) -> Result<Metadata, Error> {
    todo!();
}

pub fn extract(wasm: &[u8]) -> Result<Metadata, Error> {
    Ok(Metadata {
        name: String::new(),
    })
}

#[derive(Default)]
pub struct Manifest(Vec<(Metadata, Vec<u8>)>);

impl Manifest {
    pub fn write_to_disk(&self, dir: &Path) -> Result<(), Error> {
        todo!();
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Metadata {
    name: String,
}
