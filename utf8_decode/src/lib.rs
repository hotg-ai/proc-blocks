use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{
    ndarray::{s, ArrayView1},
    runtime_v1::*,
    BufferExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

#[macro_use]
extern crate alloc;
use alloc::string::ToString;

/// A proc block which can convert u8 bytes to utf8
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("UTF8 Decode", env!("CARGO_PKG_VERSION"));
        metadata.set_description("Decode a string from UTF-8 bytes.");
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("text");
        metadata.add_tag("nlp");
        metadata.add_tag("bytes");

        let input = TensorMetadata::new("bytes");
        input.set_description("The string as UTF-8 encoded bytes");
        let hint =
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("string");
        output.set_description("The decoded text.");
        let hint = supported_shapes(
            &[ElementType::Utf8],
            DimensionsParam::Fixed(&[1]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "bytes",
            ElementType::U8,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
            "string",
            ElementType::Utf8,
            DimensionsParam::Fixed(&[1]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_input_tensor("bytes").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "bytes".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let output = match element_type {
            ElementType::U8 => {
                let tensor = buffer
                    .view::<u8>(&dimensions)
                    .and_then(|t| t.into_dimensionality())
                    .map_err(|e| {
                        KernelError::InvalidInput(InvalidInput {
                            name: "bounding_boxes".to_string(),
                            reason: BadInputReason::InvalidValue(e.to_string()),
                        })
                    })?;
                transform(tensor)
            },
            other => {
                return Err(KernelError::Other(format!(
                "The Utf8 Decode proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        ctx.set_output_tensor(
            "string",
            TensorParam {
                element_type: ElementType::Utf8,
                dimensions: &[output.dim() as u32],
                buffer: &output.to_vec(),
            },
        );

        Ok(())
    }
}

fn transform(input: ArrayView1<u8>) -> ArrayView1<u8> {
    match input.iter().position(|&x| x == 0) {
        Some(null_terminator) => input.slice_move(s![..null_terminator]),
        None => input,
    }
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn test_for_utf8_decoding() {
        let bytes = ndarray::array![
            72_u8, 105, 44, 32, 117, 115, 101, 32, 109, 101, 32, 116, 111, 32,
            99, 111, 110, 118, 101, 114, 116, 32, 121, 111, 117, 114, 32, 117,
            56, 32, 98, 121, 116, 101, 115, 32, 116, 111, 32, 117, 116, 102,
            56, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]; // bytes encoding for "Hi, use me to convert your u8 bytes to utf8."

        let should_be = ndarray::array![
            72_u8, 105, 44, 32, 117, 115, 101, 32, 109, 101, 32, 116, 111, 32,
            99, 111, 110, 118, 101, 114, 116, 32, 121, 111, 117, 114, 32, 117,
            56, 32, 98, 121, 116, 101, 115, 32, 116, 111, 32, 117, 116, 102,
            56, 46,
        ];

        let output = transform(bytes.view());

        assert_eq!(output, should_be);
    }
}
