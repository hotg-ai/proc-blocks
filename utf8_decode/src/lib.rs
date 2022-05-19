use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{
    ndarray, runtime_v1::*, string_tensor_from_ndarray, BufferExt,
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
            ElementType::U8,
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

        match element_type {
            ElementType::U8 => {
                buffer.view::<u8>(&dimensions).map_err(|e| {
                    KernelError::InvalidInput(InvalidInput {
                        name: "bytes".to_string(),
                        reason: BadInputReason::InvalidValue(e.to_string()),
                    })
                })?;
            },
            other => {
                return Err(KernelError::Other(format!(
                "The Utf8 Decode proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        let output = transform(buffer.elements());

        ctx.set_output_tensor(
            "phrases",
            TensorParam {
                element_type: ElementType::Utf8,
                dimensions: &[output.len() as u32],
                buffer: &string_tensor_from_ndarray(&ndarray::arr1(&output)),
            },
        );

        Ok(())
    }
}

fn transform(input: &[u8]) -> Vec<String> {
    let underlying_bytes: &[u8] = input.elements();

    let mut useful_bytes = &underlying_bytes[..underlying_bytes.len()];
    if let Some(index) = underlying_bytes.iter().position(|&x| x == 0) {
        useful_bytes = &underlying_bytes[..index];
    }

    let input_text = core::str::from_utf8(useful_bytes)
        .expect("Input tensor should be valid UTF8");

    let output_text = vec![input_text.to_string()];

    output_text
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_for_utf8_decoding() {
        let bytes: Vec<u8> = "Hi, use me to convert your u8 bytes to utf8."
            .as_bytes()
            .to_vec();

        let output = transform(&bytes);
        let should_be =
            vec!["Hi, use me to convert your u8 bytes to utf8.".to_string()];

        assert_eq!(output, should_be);
    }
}
