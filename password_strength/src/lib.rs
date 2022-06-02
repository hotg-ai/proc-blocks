use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{
    ndarray::{s, ArrayView1},
    runtime_v1::*,
    BufferExt, SliceExt,
};

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
        input.set_description("The string as UTF-8 encoded bytes");
        let hint =
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
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
            "bytes",
            ElementType::U8,
            DimensionsParam::Fixed(&[0]),
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
                            name: "bytes".to_string(),
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
            "password_strength",
            TensorParam {
                element_type: ElementType::U32,
                dimensions: &[output.len() as u32],
                buffer: &output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(input: ArrayView1<u8>) -> Vec<u32> {
    let useful_bytes = match input.iter().position(|&x| x == 0) {
        Some(null_terminator) => input.slice_move(s![..null_terminator]),
        None => input,
    };
    let mut password_length: Vec<u32> = Vec::new();
    let mut word = Vec::new();

    for i in useful_bytes {
        if i == &10 {
            let text = str::from_utf8(&mut word).unwrap();
            if text.len() > 10 {
                password_length.push(0);
            } else if text.len() > 6 && text.len() <= 10 {
                password_length.push(1);
            } else {
                password_length.push(2);
            }
            word = Vec::new();
            continue;
        }
        word.push(*i);
    }
    // adding the last word length
    let text = str::from_utf8(&mut word).unwrap();
    if text.len() > 10 {
        password_length.push(0);
    } else if text.len() > 6 && text.len() <= 10 {
        password_length.push(1);
    } else {
        password_length.push(2);
    }
    return password_length;
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn test_for_utf8_decoding() {
        // aeroplane
        // bicycle
        // bird
        // boat
        // bottle
        // bus
        // car
        // cat
        // chair
        // cow
        // diningtable
        // dog
        // horse
        // motorbike
        // person
        // pottedplant
        // sheep
        // sofa
        // train
        // tv

        // bytes representation for above strings
        let bytes = ndarray::array![
            97, 101, 114, 111, 112, 108, 97, 110, 101, 10, 98, 105, 99, 121,
            99, 108, 101, 10, 98, 105, 114, 100, 10, 98, 111, 97, 116, 10, 98,
            111, 116, 116, 108, 101, 10, 98, 117, 115, 10, 99, 97, 114, 10, 99,
            97, 116, 10, 99, 104, 97, 105, 114, 10, 99, 111, 119, 10, 100, 105,
            110, 105, 110, 103, 116, 97, 98, 108, 101, 10, 100, 111, 103, 10,
            104, 111, 114, 115, 101, 10, 109, 111, 116, 111, 114, 98, 105, 107,
            101, 10, 112, 101, 114, 115, 111, 110, 10, 112, 111, 116, 116, 101,
            100, 112, 108, 97, 110, 116, 10, 115, 104, 101, 101, 112, 10, 115,
            111, 102, 97, 10, 116, 114, 97, 105, 110, 10, 116, 118
        ];
        let should_be =
            vec![1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 1, 2, 0, 2, 2, 2, 2];

        let output = transform(bytes.view());

        assert_eq!(output, should_be);
    }
}
