use crate::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError,
    },
    runtime_v1::*,
};
use hotg_rune_proc_blocks::common::element_type;

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Tensor Input", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("input");

        let arg = ArgumentMetadata::new(element_type::NAME);
        arg.set_description(element_type::DESCRIPTION);
        arg.set_default_value("f32");
        let hint = runtime_v1::interpret_as_string_in_enum(element_type::ALL);
        arg.add_hint(&hint);
        metadata.add_argument(&arg);

        let output = TensorMetadata::new("output");
        let hint = supported_shapes(
            &[
                ElementType::U8,
                ElementType::I8,
                ElementType::U16,
                ElementType::I16,
                ElementType::U32,
                ElementType::I32,
                ElementType::F32,
                ElementType::U64,
                ElementType::I64,
                ElementType::F64,
                ElementType::Utf8,
            ],
            DimensionsParam::Fixed(&[0]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&id).ok_or_else(|| {
            GraphError::Other("Unable to get the graph context".to_string())
        })?;

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("u8") => ElementType::U8,
            Some("i8") => ElementType::I8,
            Some("u16") => ElementType::U16,
            Some("i16") => ElementType::I16,
            Some("u32") => ElementType::U32,
            Some("i32") => ElementType::I32,
            Some("f32") => ElementType::F32,
            Some("u64") => ElementType::U64,
            Some("i64") => ElementType::I64,
            Some("f64") | None => ElementType::F64,
            Some("utf8") => ElementType::Utf8,
            Some(_) => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::InvalidValue(
                        "Unsupported element type".to_string(),
                    ),
                }))
            },
        };

        ctx.add_output_tensor(
            "output",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).ok_or_else(|| {
            KernelError::Other("Unable to get the kernel context".to_string())
        })?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_global_input(&id).ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: id,
                reason: BadInputReason::NotFound,
            })
        })?;

        // Dinesh, please don't look at this too closely otherwise you might
        // notice we're literally copying a tensor into WebAssembly only to
        // copy it back again 😅

        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type,
                dimensions: &dimensions,
                buffer: &buffer,
            },
        );

        Ok(())
    }
}
