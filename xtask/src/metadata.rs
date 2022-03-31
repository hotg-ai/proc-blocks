use self::rune_v1::{RuneV1, RuneV1Data};
use crate::CompiledModule;
use anyhow::{Context, Error};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, fs::File, num::NonZeroUsize, path::Path, sync::Mutex,
};
use wasmtime::{Engine, Linker, Module, Store};

wit_bindgen_wasmtime::export!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_wasmtime::import!("../wit-files/rune/rune-v1.wit");

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
    let engine = Engine::default();

    tracing::debug!("Loading the WebAssembly module");

    let module = Module::new(&engine, serialized)
        .context("Unable to instantiate the module")?;
    let mut store = Store::new(&engine, State::default());

    tracing::debug!("Setting up the host functions");

    let mut linker = Linker::new(&engine);
    runtime_v1::add_to_linker(&mut linker, |state: &mut State| {
        (&mut state.runtime, &mut state.tables)
    })
    .context("Unable to register the host functions")?;

    tracing::debug!("Instantiating the WebAssembly module");

    let (rune, _) = RuneV1::instantiate(
        &mut store,
        &module,
        &mut linker,
        |state: &mut State| &mut state.rune_v1_data,
    )
    .context("Unable to instantiate the WebAssembly module")?;

    tracing::debug!("Running the start() function");

    rune.start(&mut store)
        .context("Unable to run the WebAssembly module's start() function")?;

    store
        .data_mut()
        .runtime
        .node
        .take()
        .context("The WebAssembly module didn't register any metadata")
}

#[derive(Default)]
struct State {
    runtime: Runtime,
    tables: runtime_v1::RuntimeV1Tables<Runtime>,
    rune_v1_data: RuneV1Data,
}

#[derive(Default)]
pub struct Manifest {
    metadata: HashMap<String, Metadata>,
    serialized: HashMap<String, Vec<u8>>,
}

impl Manifest {
    pub fn write_to_disk(&self, dir: &Path) -> Result<(), Error> {
        let _span = tracing::info_span!("Saving");

        std::fs::create_dir_all(dir).with_context(|| {
            format!("Unable to create the \"{}\" directory", dir.display())
        })?;

        for (name, wasm) in &self.serialized {
            let filename = dir.join(&name);
            std::fs::write(&filename, wasm).with_context(|| {
                format!("Unable to save to \"{}\"", filename.display())
            })?;
        }

        save_json(dir.join("metadata.json"), &self.metadata)
            .context("Unable to save the metadata")?;

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

    let f = File::create(path).with_context(|| {
        format!("Unable to open \"{}\" for writing", path.display())
    })?;

    serde_json::to_writer_pretty(f, &value)?;

    Ok(())
}

#[derive(Default)]
struct Runtime {
    node: Option<Metadata>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Metadata {
    name: String,
    version: String,
    description: Option<String>,
    repository: Option<String>,
    homepage: Option<String>,
    tags: Vec<String>,
    arguments: Vec<ArgumentMetadata>,
    inputs: Vec<TensorMetadata>,
    outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ArgumentMetadata {
    name: String,
    description: Option<String>,
    default_value: Option<String>,
    hints: Vec<ArgumentHint>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TensorMetadata {
    name: String,
    description: Option<String>,
    hints: Vec<TensorHint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
enum TensorHint {
    DisplayAs(String),
    SupportedShape {
        accepted_element_types: Vec<ElementType>,
        dimensions: Dimensions,
    },
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ElementType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    Utf8,
}

impl From<runtime_v1::ElementType> for ElementType {
    fn from(e: runtime_v1::ElementType) -> Self {
        match e {
            runtime_v1::ElementType::U8 => ElementType::U8,
            runtime_v1::ElementType::I8 => ElementType::I8,
            runtime_v1::ElementType::U16 => ElementType::U16,
            runtime_v1::ElementType::I16 => ElementType::I16,
            runtime_v1::ElementType::U32 => ElementType::U32,
            runtime_v1::ElementType::I32 => ElementType::I32,
            runtime_v1::ElementType::F32 => ElementType::F32,
            runtime_v1::ElementType::I64 => ElementType::I64,
            runtime_v1::ElementType::U64 => ElementType::U64,
            runtime_v1::ElementType::F64 => ElementType::F64,
            runtime_v1::ElementType::Utf8 => ElementType::Utf8,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
enum Dimensions {
    Dynamic,
    Fixed(Vec<Dimension>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
enum Dimension {
    Fixed(NonZeroUsize),
    Dynamic,
}

impl From<runtime_v1::Dimensions<'_>> for Dimensions {
    fn from(d: runtime_v1::Dimensions<'_>) -> Self {
        match d {
            runtime_v1::Dimensions::Dynamic => Dimensions::Dynamic,
            runtime_v1::Dimensions::Fixed(dims) => Dimensions::Fixed(
                dims.iter()
                    .map(|d| match NonZeroUsize::new(d.get() as usize) {
                        Some(d) => Dimension::Fixed(d),
                        None => Dimension::Dynamic,
                    })
                    .collect(),
            ),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
enum ArgumentHint {
    NonNegativeNumber,
    StringEnum(Vec<String>),
    NumberInRange {
        max: String,
        min: String,
    },
    #[serde(with = "ArgumentTypeRepr")]
    SupportedArgumentType(runtime_v1::ArgumentType),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[serde(remote = "runtime_v1::ArgumentType")]
enum ArgumentTypeRepr {
    UnsignedInteger,
    Integer,
    Float,
    String,
    LongString,
}

#[derive(Debug)]
struct GraphContext;

#[derive(Debug)]
struct KernelContext;

impl runtime_v1::RuntimeV1 for Runtime {
    type ArgumentHint = ArgumentHint;
    type ArgumentMetadata = Mutex<ArgumentMetadata>;
    type GraphContext = GraphContext;
    type KernelContext = KernelContext;
    type Metadata = Mutex<Metadata>;
    type TensorHint = TensorHint;
    type TensorMetadata = Mutex<TensorMetadata>;

    fn metadata_new(&mut self, name: &str, version: &str) -> Self::Metadata {
        Mutex::new(Metadata {
            name: name.to_string(),
            version: version.to_string(),
            ..Default::default()
        })
    }

    fn metadata_set_description(
        &mut self,
        self_: &Self::Metadata,
        description: &str,
    ) {
        self_.lock().unwrap().description = Some(description.to_string());
    }

    fn metadata_set_repository(&mut self, self_: &Self::Metadata, url: &str) {
        self_.lock().unwrap().repository = Some(url.to_string());
    }

    fn metadata_set_homepage(&mut self, self_: &Self::Metadata, url: &str) {
        self_.lock().unwrap().homepage = Some(url.to_string());
    }

    fn metadata_add_tag(&mut self, self_: &Self::Metadata, tag: &str) {
        self_.lock().unwrap().tags.push(tag.to_string());
    }

    fn metadata_add_argument(
        &mut self,
        self_: &Self::Metadata,
        arg: &Self::ArgumentMetadata,
    ) {
        self_
            .lock()
            .unwrap()
            .arguments
            .push(arg.lock().unwrap().clone());
    }

    fn metadata_add_input(
        &mut self,
        self_: &Self::Metadata,
        metadata: &Self::TensorMetadata,
    ) {
        self_
            .lock()
            .unwrap()
            .inputs
            .push(metadata.lock().unwrap().clone());
    }

    fn metadata_add_output(
        &mut self,
        self_: &Self::Metadata,
        metadata: &Self::TensorMetadata,
    ) {
        self_
            .lock()
            .unwrap()
            .outputs
            .push(metadata.lock().unwrap().clone());
    }

    fn argument_metadata_new(&mut self, name: &str) -> Self::ArgumentMetadata {
        Mutex::new(ArgumentMetadata {
            name: name.to_string(),
            ..Default::default()
        })
    }

    fn argument_metadata_set_description(
        &mut self,
        self_: &Self::ArgumentMetadata,
        description: &str,
    ) {
        self_.lock().unwrap().description = Some(description.to_string());
    }

    fn argument_metadata_set_default_value(
        &mut self,
        self_: &Self::ArgumentMetadata,
        default_value: &str,
    ) {
        self_.lock().unwrap().default_value = Some(default_value.to_string());
    }

    fn argument_metadata_add_hint(
        &mut self,
        self_: &Self::ArgumentMetadata,
        hint: &Self::ArgumentHint,
    ) {
        self_.lock().unwrap().hints.push(hint.clone());
    }

    fn tensor_metadata_new(&mut self, name: &str) -> Self::TensorMetadata {
        Mutex::new(TensorMetadata {
            name: name.to_string(),
            ..Default::default()
        })
    }

    fn tensor_metadata_set_description(
        &mut self,
        self_: &Self::TensorMetadata,
        description: &str,
    ) {
        self_.lock().unwrap().description = Some(description.to_string());
    }

    fn tensor_metadata_add_hint(
        &mut self,
        self_: &Self::TensorMetadata,
        hint: &Self::TensorHint,
    ) {
        self_.lock().unwrap().hints.push(hint.clone());
    }

    fn interpret_as_image(&mut self) -> Self::TensorHint {
        TensorHint::DisplayAs("image".to_string())
    }

    fn interpret_as_audio(&mut self) -> Self::TensorHint {
        TensorHint::DisplayAs("audio".to_string())
    }

    fn supported_shapes(
        &mut self,
        supported_element_type: Vec<runtime_v1::ElementType>,
        dimensions: runtime_v1::Dimensions<'_>,
    ) -> Self::TensorHint {
        TensorHint::SupportedShape {
            accepted_element_types: supported_element_type
                .into_iter()
                .map(ElementType::from)
                .collect(),
            dimensions: dimensions.into(),
        }
    }

    fn interpret_as_number_in_range(
        &mut self,
        min: &str,
        max: &str,
    ) -> Self::ArgumentHint {
        ArgumentHint::NumberInRange {
            min: min.to_string(),
            max: max.to_string(),
        }
    }

    fn interpret_as_string_in_enum(
        &mut self,
        string_enum: Vec<&str>,
    ) -> Self::ArgumentHint {
        ArgumentHint::StringEnum(
            string_enum.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn non_negative_number(&mut self) -> Self::ArgumentHint {
        ArgumentHint::NonNegativeNumber
    }

    fn supported_argument_type(
        &mut self,
        hint: runtime_v1::ArgumentType,
    ) -> Self::ArgumentHint {
        ArgumentHint::SupportedArgumentType(hint)
    }

    fn register_node(&mut self, metadata: &Self::Metadata) {
        self.node = Some(metadata.lock().unwrap().clone());
    }

    fn graph_context_current(&mut self) -> Option<Self::GraphContext> { None }

    fn graph_context_get_argument(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
    ) -> Option<String> {
        unimplemented!()
    }

    fn graph_context_add_input_tensor(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::Dimensions<'_>,
    ) {
        unimplemented!()
    }

    fn graph_context_add_output_tensor(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::Dimensions<'_>,
    ) {
        unimplemented!()
    }

    fn kernel_context_current(&mut self) -> Option<Self::KernelContext> { None }

    fn kernel_context_get_argument(
        &mut self,
        self_: &Self::KernelContext,
        name: &str,
    ) -> Option<String> {
        unimplemented!()
    }

    fn kernel_context_get_input_tensor(
        &mut self,
        self_: &Self::KernelContext,
        name: &str,
    ) -> Option<runtime_v1::TensorResult> {
        unimplemented!()
    }

    fn kernel_context_set_output_tensor(
        &mut self,
        self_: &Self::KernelContext,
        name: &str,
        tensor: runtime_v1::TensorParam<'_>,
    ) {
        unimplemented!()
    }
}
