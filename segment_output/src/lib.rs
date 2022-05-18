use std::convert::TryInto;

use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};

use hotg_rune_proc_blocks::{ndarray::s, runtime_v1::*, BufferExt, SliceExt};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// A proc-block which takes a rank 4 `tensor` as input, whose dimension is of
/// this form `[1, x, y, z]`.
///
/// It will return:
/// 1. a 2-d `tensor` after performing argmax along the axis-3 of the tensor
/// 2. a 1-d `tensor` which a `set` of all the number present in the above 2-d
///    `tensor`
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Segment Output", env!("CARGO_PKG_VERSION"));
        metadata.set_description("Useful in image segmentation. A proc-block which takes a rank 4 tensor as input, whose dimension is of this form `[1, rows, columns, confidence]`.");
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("image");
        metadata.add_tag("segmentation");

        let input = TensorMetadata::new("image");
        input.set_description("An image-like tensor with the dimensions, `[1, rows, columns, category_confidence]`. Each \"pixel\" is associated with a set of confidence values, where each value indicates how confident the model is that the pixel is in that category.");
        let hint = supported_shapes(
            &[ElementType::F32],
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let segmentation_map = TensorMetadata::new("segmentation_map");
        segmentation_map.set_description("An image-like tensor where each pixel contains the index of the category with the highest confidence level.");
        let hint = supported_shapes(
            &[ElementType::U32],
            DimensionsParam::Fixed(&[0, 0]),
        );
        segmentation_map.add_hint(&hint);
        metadata.add_output(&segmentation_map);

        let indices = TensorMetadata::new("indices");
        indices.set_description("The categories used in `segmentation_map`.");
        let hint =
            supported_shapes(&[ElementType::U32], DimensionsParam::Fixed(&[0]));
        indices.add_hint(&hint);
        metadata.add_output(&indices);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "input",
            ElementType::F32,
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );

        ctx.add_output_tensor(
            "segmentation_map",
            ElementType::U32,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_output_tensor(
            "indices",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx =
            KernelContext::for_node(&id).ok_or(KernelError::MissingContext)?;
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

        let (segmented_map, indices) = match element_type {
            ElementType::F32 => transform(buffer, &dimensions),
            other => {
                return Err(KernelError::Other(format!(
                "The softmax proc-block only accepts f32 or f64 tensors, found {:?}",
                other,
                )))
            },
        };

        ctx.set_output_tensor(
            "segmentation_map",
            TensorParam {
                element_type,
                dimensions: &[dimensions[1], dimensions[2]],
                buffer: &segmented_map.as_bytes(),
            },
        );

        ctx.set_output_tensor(
            "indices",
            TensorParam {
                element_type,
                dimensions: &[indices.len().try_into().unwrap()],
                buffer: &indices.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(input: Vec<u8>, dimension: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let dim = dimension;
    // todo: check for the dimension
    // match input.rank() {
    //     4 => {
    //         if dim[0] != 1 {
    //             panic!("the first dimension should be 1")
    //         }
    //     },
    //     _ => panic!("it only accept a rank 4 tensor"),
    // }

    let input = input.view::<f32>(&dim).unwrap();
    let mut vec_2d: Vec<Vec<u32>> = Vec::new();

    let rows = dim[1] as usize;
    let columns = dim[2] as usize;

    let mut label_index = Vec::new();

    for i in 0..rows {
        vec_2d.push(vec![]);
        for j in 0..columns {
            let val = input.slice(s![0 as usize, i, j]);

            let (index, _) =
                val.iter().enumerate().fold((0, 0.0), |max, (ind, &val)| {
                    if val > max.1 {
                        (ind, val)
                    } else {
                        max
                    }
                }); // Doing argmax over the array
            vec_2d[i].push(index as u32);
            label_index.push(index as u32);
        }
    }

    (
        vec_2d.iter().flat_map(|arr| arr.iter()).cloned().collect(),
        label_index,
    )
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let input = [
            1.7611206, -0.824405, 3.3042068, 4.1308413, 3.8263698, 13.207806,
            3.4352894, 4.6627636, 4.464175, 1.7611206, 8.24405, 3.3042068,
            1.7611206, -0.824405, 3.3042068, 4.1308413, 3.8263698, 13.207806,
            3.4352894, 4.6627636, 4.464175, 1.7611206, 8.24405, 3.3042068,
            1.7611206, -0.824405, 3.3042068, 4.1308413, 3.8263698, 13.207806,
            3.4352894, 4.6627636, 4.464175, 1.7611206, 8.24405, 3.3042068,
            1.7611206, -0.824405, 3.3042068, 4.1308413, 3.8263698, 13.207806,
            3.4352894, 4.6627636, 4.464175, 1.7611206, 8.24405, 3.3042068,
            1.7611206, -0.824405, 3.3042068, 4.1308413, 3.8263698, 13.207806,
            3.4352894, 4.6627636, 4.464175, 1.7611206, 8.24405, 3.3042068,
        ]
        .as_bytes()
        .to_vec();

        let output = transform(input, &[1, 5, 4, 3]);
        let should_be: Vec<u32> =
            vec![2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1];
        // let should_be: Tensor<u32> = should_be.into();
        let label_index: Vec<u32> = vec![1, 2];
        assert_eq!(output, (should_be, label_index));
    }

    // #[test]
    // #[should_panic = "it only accept a rank 4 tensor"]
    // fn not_rank_4_tensor() {
    //     let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].as_bytes().to_vec();
    // // a rank 3 array     let _output = transform(input, &[1, 2, 3]);
    // }

    // #[test]
    // #[should_panic = "the first dimension should be 1"]
    // fn first_dimension_not_1() {
    //     let input = [[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 6.0]]]]; //
    // [3,1,1,2] dimension array     let input: Tensor<f32> = input.into();
    //     let mut argmax = SegmentOutput::default();
    //     let _output = argmax.transform(input);
    // }
}
