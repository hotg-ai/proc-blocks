use crate::runtime::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError, ProcBlockV1,
    },
    runtime_v1::LogMetadata,
};
use anyhow::{Context, Error};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};
use wasmer::{ImportObject, Module, Store, WasmerEnv};

wit_bindgen_wasmer::export!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_wasmer::import!("../wit-files/rune/proc-block-v1.wit");

pub struct Runtime {
    rune: ProcBlockV1,
    shared: Arc<Mutex<Shared>>,
}

impl Runtime {
    #[tracing::instrument(skip(wasm))]
    pub fn load(wasm: &[u8]) -> Result<Self, Error> {
        tracing::debug!("Loading the WebAssembly module");

        let mut store = Store::default();
        let module = Module::new(&store, wasm)
            .context("Unable to instantiate the module")?;

        tracing::debug!("Setting up the host functions");

        let mut imports = ImportObject::default();
        let shared = Arc::new(Mutex::new(Shared::default()));
        runtime_v1::add_to_imports(
            &store,
            &mut imports,
            RuntimeV1(shared.clone()),
        );

        tracing::debug!("Instantiating the WebAssembly module");

        let (rune, _) =
            ProcBlockV1::instantiate(&mut store, &module, &mut imports)
                .context("Unable to instantiate the WebAssembly module")?;

        Ok(Runtime { rune, shared })
    }

    #[tracing::instrument(skip(self))]
    pub fn metadata(&mut self) -> Result<Metadata, Error> {
        tracing::debug!("Running the register_metadata() function");

        self.rune.register_metadata().context(
            "Unable to run the WebAssembly module's register_metadata() function",
        )?;

        let mut shared = self.shared.lock().unwrap();
        let metadata = std::mem::take(&mut shared.metadata);

        Ok(metadata)
    }

    #[tracing::instrument(skip(self, args))]
    pub fn graph(
        &mut self,
        args: HashMap<String, String>,
    ) -> Result<NodeInfo, Error> {
        let mut shared = self.shared.lock().unwrap();
        shared.args = args;
        drop(shared);

        self.rune
            .graph("")
            .context("Unable to call the graph() function")??;

        let mut shared = self.shared.lock().unwrap();
        Ok(std::mem::take(&mut shared.node))
    }
}

#[derive(Default, Clone, WasmerEnv)]
struct RuntimeV1(Arc<Mutex<Shared>>);

#[derive(Default, Clone, WasmerEnv)]
struct Shared {
    args: HashMap<String, String>,
    metadata: Metadata,
    node: NodeInfo,
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

impl Display for ElementType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ElementType::U8 => write!(f, "u8"),
            ElementType::I8 => write!(f, "i8"),
            ElementType::U16 => write!(f, "u16"),
            ElementType::I16 => write!(f, "i16"),
            ElementType::U32 => write!(f, "u32"),
            ElementType::I32 => write!(f, "i32"),
            ElementType::F32 => write!(f, "f32"),
            ElementType::U64 => write!(f, "u64"),
            ElementType::I64 => write!(f, "i64"),
            ElementType::F64 => write!(f, "f64"),
            ElementType::Utf8 => write!(f, "utf-8"),
        }
    }
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

impl Display for Dimension {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Dimension::Fixed(fixed) => fixed.fmt(f),
            Dimension::Dynamic => "*".fmt(f),
        }
    }
}

impl From<runtime_v1::DimensionsParam<'_>> for Dimensions {
    fn from(d: runtime_v1::DimensionsParam<'_>) -> Self {
        match d {
            runtime_v1::DimensionsParam::Dynamic => Dimensions::Dynamic,
            runtime_v1::DimensionsParam::Fixed(dims) => Dimensions::Fixed(
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

impl runtime_v1::RuntimeV1 for RuntimeV1 {
    type ArgumentHint = ArgumentHint;
    type ArgumentMetadata = Mutex<ArgumentMetadata>;
    type GraphContext = ();
    type KernelContext = ();
    type Metadata = Mutex<Metadata>;
    type Model = ();
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
        dimensions: runtime_v1::DimensionsParam<'_>,
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
        self.0.lock().unwrap().metadata = metadata.lock().unwrap().clone();
    }

    fn graph_context_for_node(&mut self, _name: &str) -> Option<()> { Some(()) }

    fn graph_context_get_argument(
        &mut self,
        _: &Self::GraphContext,
        name: &str,
    ) -> Option<String> {
        self.0.lock().unwrap().args.get(name).cloned()
    }

    fn graph_context_add_input_tensor(
        &mut self,
        _: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::DimensionsParam<'_>,
    ) {
        let mut shared = self.0.lock().unwrap();
        shared.node.inputs.push(TensorInfo {
            name: name.to_string(),
            element_type: element_type.into(),
            dimensions: dimensions.into(),
        })
    }

    fn graph_context_add_output_tensor(
        &mut self,
        _: &Self::GraphContext,
        name: &str,
        element_type: runtime_v1::ElementType,
        dimensions: runtime_v1::DimensionsParam<'_>,
    ) {
        let mut shared = self.0.lock().unwrap();
        shared.node.outputs.push(TensorInfo {
            name: name.to_string(),
            element_type: element_type.into(),
            dimensions: dimensions.into(),
        })
    }

    fn kernel_context_for_node(
        &mut self,
        _name: &str,
    ) -> Option<Self::KernelContext> {
        Some(())
    }

    fn kernel_context_get_argument(
        &mut self,
        _: &Self::KernelContext,
        name: &str,
    ) -> Option<String> {
        self.0.lock().unwrap().args.get(name).cloned()
    }

    fn kernel_context_get_input_tensor(
        &mut self,
        _: &Self::KernelContext,
        _name: &str,
    ) -> Option<runtime_v1::TensorResult> {
        unimplemented!()
    }

    fn kernel_context_set_output_tensor(
        &mut self,
        _: &Self::KernelContext,
        _name: &str,
        _tensor: runtime_v1::TensorParam<'_>,
    ) {
        unimplemented!()
    }

    fn is_enabled(&mut self, _metadata: LogMetadata<'_>) -> bool { true }

    fn log(
        &mut self,
        metadata: LogMetadata<'_>,
        message: &str,
        data: runtime_v1::LogValueMap<'_>,
    ) {
        tracing::info!(?metadata, ?data, message);
    }

    fn kernel_context_get_global_input(
        &mut self,
        _: &Self::KernelContext,
        _name: &str,
    ) -> Option<runtime_v1::TensorResult> {
        todo!()
    }

    fn kernel_context_set_global_output(
        &mut self,
        _: &Self::KernelContext,
        _name: &str,
        _tensor: runtime_v1::TensorParam<'_>,
    ) {
        todo!()
    }

    fn model_load(
        &mut self,
        _model_format: &str,
        _model: &[u8],
        _arguments: Vec<(&str, &str)>,
    ) -> Result<Self::Model, runtime_v1::ModelLoadError> {
        todo!()
    }

    fn model_infer(
        &mut self,
        _self_: &Self::Model,
        _inputs: Vec<runtime_v1::TensorParam<'_>>,
    ) -> Result<Vec<runtime_v1::TensorResult>, runtime_v1::ModelInferError>
    {
        todo!()
    }

    fn model_inputs(&mut self, _self_: &Self::Model) -> Vec<runtime_v1::Shape> {
        todo!()
    }

    fn model_outputs(
        &mut self,
        _self_: &Self::Model,
    ) -> Vec<runtime_v1::Shape> {
        todo!()
    }
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::InvalidArgument(a) => a.fmt(f),
            GraphError::MissingContext => {
                write!(f, "The context wasn't passed in")
            },
            GraphError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GraphError::InvalidArgument(a) => a.source(),
            GraphError::MissingContext | GraphError::Other(_) => None,
        }
    }
}

impl Display for KernelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::InvalidArgument(a) => a.fmt(f),
            KernelError::InvalidInput(i) => i.fmt(f),
            KernelError::MissingContext => {
                write!(f, "The context wasn't passed in")
            },
            KernelError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for KernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KernelError::InvalidArgument(a) => a.source(),
            KernelError::InvalidInput(i) => i.source(),
            KernelError::MissingContext | KernelError::Other(_) => None,
        }
    }
}

impl Display for InvalidInput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "The \"{}\" input tensor was invalid", self.name)
    }
}

impl std::error::Error for InvalidInput {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.reason)
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

impl Display for BadInputReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BadInputReason::NotFound => {
                write!(f, "The input tensor wasn't provided")
            },
            BadInputReason::InvalidValue(reason) => {
                write!(f, "{}", reason)
            },
            BadInputReason::UnsupportedShape => {
                write!(f, "Unsupported shape")
            },
            BadInputReason::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for BadInputReason {}
