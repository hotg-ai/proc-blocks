use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{prelude::*, runtime_v1::*};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Tensor Input", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("input");
        metadata.add_tag("raw");

        let element_type = ArgumentMetadata::element_type();
        metadata.add_argument(&element_type);

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

        let element_type: ElementType =
            ctx.parse_argument_with_default("element_type", ElementType::F32)?;

        ctx.add_input_tensor("input", element_type, DimensionsParam::Dynamic);
        ctx.add_output_tensor("output", element_type, DimensionsParam::Dynamic);

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
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "input".to_string(),
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

impl ContextErrorExt for GraphError {
    type InvalidArgument = InvalidArgument;

    fn invalid_argument(inner: InvalidArgument) -> Self {
        GraphError::InvalidArgument(inner)
    }
}

impl InvalidArgumentExt for InvalidArgument {
    fn other(name: &str, msg: impl std::fmt::Display) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::Other(msg.to_string()),
        }
    }

    fn invalid_value(name: &str, error: impl std::fmt::Display) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::InvalidValue(error.to_string()),
        }
    }

    fn not_found(name: &str) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::NotFound,
        }
    }
}
