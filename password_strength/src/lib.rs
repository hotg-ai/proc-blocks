use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};

use std::str;

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
        metadata.add_tag("bytes");
        metadata.add_tag("string");

        let input = TensorMetadata::new("bytes");
        input.set_description("string");
        let hint =
            supported_shapes(&[ElementType::Utf8], DimensionsParam::Dynamic);
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("password_strength");
        output.set_description("Label for Password strength");
        let hint =
            supported_shapes(&[ElementType::U32], DimensionsParam::Fixed(&[0]));
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "string",
            ElementType::Utf8,
            DimensionsParam::Dynamic,
        );

        ctx.add_output_tensor(
            "password_strength",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
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
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "input".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let words = match element_type {
            ElementType::Utf8 => buffer
                .strings()
                .map_err(|e| KernelError::Other(e.to_string()))?,
            other => {
                return Err(KernelError::Other(format!(
                "The Parse proc-block only accepts Utf8 tensors, found {:?}",
                other,
            )))
            },
        };

        let output = transform(words);

        ctx.set_output_tensor(
            "password_strength",
            TensorParam {
                element_type: ElementType::U32,
                dimensions: &dimensions,
                buffer: &output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(input: Vec<&str>) -> Vec<u32> {
    let mut password_length: Vec<u32> = Vec::new();

    for i in input {
        println!("{:?}", &i);
        if &i[i.len() - 1..] == String::from('\n').as_str() {
            if i.len() > 11 {
                password_length.push(0);
            } else if i.len() > 7 && i.len() <= 11 {
                password_length.push(1);
            } else {
                password_length.push(2);
            }
            continue;
        }
    }

    return password_length;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_for_utf8_decoding() {
        let string = vec![
            "aeroplane\n",
            "bicycle\n",
            "bird\n",
            "boat\n",
            "bottle\n",
            "bus\n",
            "car\n",
            "cat\n",
            "chair\n",
            "cow\n",
            "diningtable\n",
            "dog\n",
            "horse\n",
            "motorbike\n",
            "person\n",
            "pottedplant\n",
            "sheep\n",
            "sofa\n",
            "train\n",
            "tv\n",
        ];

        let should_be =
            vec![1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 1, 2, 0, 2, 2, 2, 2];

        let output = transform(string);

        assert_eq!(output, should_be);
    }
}
