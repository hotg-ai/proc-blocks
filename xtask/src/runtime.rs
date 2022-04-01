use crate::runtime::rune_v1::{
    BadArgumentReason, GraphError, InvalidArgument, KernelError,
};

use self::rune_v1::{RuneV1, RuneV1Data};
use anyhow::{Context, Error};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};
use wasmtime::{Engine, Linker, Module, Store};

wit_bindgen_wasmtime::export!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_wasmtime::import!("../wit-files/rune/rune-v1.wit");

pub struct Runtime {
    rune: RuneV1<State>,
    store: Store<State>,
}

impl Runtime {
    #[tracing::instrument(skip(wasm))]
    pub fn load(wasm: &[u8]) -> Result<Self, Error> {
        let engine = Engine::default();

        tracing::debug!("Loading the WebAssembly module");

        let module = Module::new(&engine, wasm)
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

        Ok(Runtime { rune, store })
    }

    #[tracing::instrument(skip(self))]
    pub fn metadata(&mut self) -> Result<Metadata, Error> {
        tracing::debug!("Running the start() function");

        self.rune.start(&mut self.store).context(
            "Unable to run the WebAssembly module's start() function",
        )?;

        self.store
            .data_mut()
            .runtime
            .node
            .take()
            .context("The WebAssembly module didn't register any metadata")
    }

    #[tracing::instrument(skip(self, args))]
    pub fn graph(
        &mut self,
        args: HashMap<String, String>,
    ) -> Result<NodeInfo, Error> {
        let ctx = GraphContext::new(args);
        self.store.data_mut().runtime.graph_ctx =
            Some(Arc::new(Mutex::new(ctx)));

        self.rune
            .graph(&mut self.store)
            .context("Unable to call the graph() function")?
            .context("Unable to determine the node's inputs and outputs")?;

        let ctx = self.store.data_mut().runtime.graph_ctx.take().unwrap();
        let ctx = ctx.lock().unwrap();
        Ok(ctx.node.clone())
    }
}

#[derive(Default)]
struct State {
    runtime: RuntimeV1,
    tables: runtime_v1::RuntimeV1Tables<RuntimeV1>,
    rune_v1_data: RuneV1Data,
}

#[derive(Default)]
struct RuntimeV1 {
    node: Option<Metadata>,
    graph_ctx: Option<Arc<Mutex<GraphContext>>>,
    kernel_ctx: Option<Arc<Mutex<KernelContext>>>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub tags: Vec<String>,
    pub arguments: Vec<ArgumentMetadata>,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ArgumentMetadata {
    pub name: String,
    pub description: Option<String>,
    pub default_value: Option<String>,
    pub hints: Vec<ArgumentHint>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TensorMetadata {
    pub name: String,
    pub description: Option<String>,
    pub hints: Vec<TensorHint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
pub enum TensorHint {
    DisplayAs(String),
    SupportedShape {
        accepted_element_types: Vec<ElementType>,
        dimensions: Dimensions,
    },
}

#[derive(
    Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ElementType {
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
pub enum Dimensions {
    Dynamic,
    Fixed(Vec<Dimension>),
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type", content = "value")]
pub enum Dimension {
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
pub enum ArgumentHint {
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
pub struct GraphContext {
    args: HashMap<String, String>,
    node: NodeInfo,
}

impl GraphContext {
    pub fn new(args: HashMap<String, String>) -> Self {
        GraphContext {
            args,
            node: NodeInfo::default(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct NodeInfo {
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TensorInfo {
    pub name: String,
    pub element_type: ElementType,
    pub dimensions: Dimensions,
}

#[derive(Debug)]
pub struct KernelContext {
    args: HashMap<String, String>,
}

impl runtime_v1::RuntimeV1 for RuntimeV1 {
    type ArgumentHint = ArgumentHint;
    type ArgumentMetadata = Mutex<ArgumentMetadata>;
    type GraphContext = Arc<Mutex<GraphContext>>;
    type KernelContext = Arc<Mutex<KernelContext>>;
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

    fn graph_context_current(&mut self) -> Option<Self::GraphContext> {
        self.graph_ctx.clone()
    }

    fn graph_context_get_argument(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
    ) -> Option<String> {
        self_.lock().unwrap().args.get(name).cloned()
    }

    fn graph_context_add_input_tensor(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::Dimensions<'_>,
    ) {
        self_.lock().unwrap().node.inputs.push(TensorInfo {
            name: name.to_string(),
            element_type: element_type.into(),
            dimensions: dimensions.into(),
        })
    }

    fn graph_context_add_output_tensor(
        &mut self,
        self_: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::Dimensions<'_>,
    ) {
        self_.lock().unwrap().node.outputs.push(TensorInfo {
            name: name.to_string(),
            element_type: element_type.into(),
            dimensions: dimensions.into(),
        })
    }

    fn kernel_context_current(&mut self) -> Option<Self::KernelContext> {
        self.kernel_ctx.clone()
    }

    fn kernel_context_get_argument(
        &mut self,
        self_: &Self::KernelContext,
        name: &str,
    ) -> Option<String> {
        self_.lock().unwrap().args.get(name).cloned()
    }

    fn kernel_context_get_input_tensor(
        &mut self,
        _self_: &Self::KernelContext,
        _name: &str,
    ) -> Option<runtime_v1::TensorResult> {
        unimplemented!()
    }

    fn kernel_context_set_output_tensor(
        &mut self,
        _self_: &Self::KernelContext,
        _name: &str,
        _tensor: runtime_v1::TensorParam<'_>,
    ) {
        unimplemented!()
    }
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::InvalidArgument(a) => a.fmt(f),
            GraphError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GraphError::InvalidArgument(a) => a.source(),
            GraphError::Other(_) => None,
        }
    }
}

impl Display for KernelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::InvalidArgument(a) => a.fmt(f),
            KernelError::MissingInput(name) => {
                write!(f, "The \"{}\" input wasn't provided", name)
            },
            KernelError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for KernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KernelError::InvalidArgument(a) => a.source(),
            KernelError::MissingInput(_) => None,
            KernelError::Other(_) => None,
        }
    }
}

impl Display for InvalidArgument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "The \"{}\" argument was invalid", self.name)
    }
}

impl std::error::Error for InvalidArgument {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.reason)
    }
}

impl Display for BadArgumentReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BadArgumentReason::NotFound => {
                write!(f, "The argument wasn't provided")
            },
            BadArgumentReason::InvalidValue(reason) => {
                write!(f, "{}", reason)
            },
            BadArgumentReason::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for BadArgumentReason {}
