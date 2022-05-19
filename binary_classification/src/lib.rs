use std::fmt::Display;

use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    runtime_v1::{self, *},
    BufferExt, SliceExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

#[macro_use]
extern crate alloc;

use alloc::vec::Vec;
use proc_block_v1::{BadArgumentReason, InvalidArgument};

/// A proc-block which takes a rank 1 `tensor` as input, return 1 if value
/// inside the tensor is greater than 1 otherwise 0.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Binary Classification", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "Classify each element in a tensor depending on whether they are above or below a certain threshold.",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("classify");

        let threshold = ArgumentMetadata::new("threshold");
        threshold.set_default_value("0.5");
        threshold.set_description("The classification threshold.");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        threshold.add_hint(&hint);
        metadata.add_argument(&threshold);

        let input = TensorMetadata::new("input");
        input.set_description("The numbers to classify");
        let hint =
            supported_shapes(&[ElementType::F32], DimensionsParam::Fixed(&[0]));
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("classified");
        output.set_description("A tensor of `1`'s and `0`'s, where `1` indicates an element was above the `threshold` and `0` means it was below.");
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
            "input",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
        );
        ctx.add_output_tensor(
            "classified",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let threshold = get_threshold(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

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

        let output = match element_type {
            ElementType::F32 =>{
                buffer.view::<f32>(&dimensions)
                .map_err(|e| KernelError::InvalidInput(InvalidInput{ name: "bounding_boxes".to_string(), reason: BadInputReason::InvalidValue(e.to_string()) }))?;
                transform(buffer.elements(), threshold)
            }
            other => {
                return Err(KernelError::Other(format!(
                "The Object Filter proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        ctx.set_output_tensor(
            "normalized",
            TensorParam {
                element_type: ElementType::U32,
                dimensions: &dimensions,
                buffer: &output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn get_threshold(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<f32, InvalidArgument> {
    get_argument("threshold")
        .ok_or_else(|| InvalidArgument::not_found("threshold"))?
        .parse::<f32>()
        .map_err(|e| InvalidArgument::invalid_value("threshold", e))
}

impl InvalidArgument {
    fn not_found(name: impl Into<String>) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::NotFound,
        }
    }

    fn invalid_value(name: impl Into<String>, reason: impl Display) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::InvalidValue(reason.to_string()),
        }
    }
}

fn transform(input: &[f32], threshold: f32) -> Vec<u32> {
    // let value = input.into();
    let mut label: u32 = 0;
    if input > &[threshold] {
        label = 1
    }
    let v: Vec<u32> = vec![label];
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_classification() {
        let v = vec![0.7];
        let output = transform(&v, 0.5);
        let should_be = vec![1];
        assert_eq!(output, should_be);
    }
}
