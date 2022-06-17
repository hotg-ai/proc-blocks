use std::collections::BTreeSet;

use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementType, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{s, Array1, Array2, ArrayView4},
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: SegmentOutput,
}

fn metadata() -> Metadata {
    Metadata::new("Segment Output", env!("CARGO_PKG_VERSION"))
        .with_description("Useful in image segmentation. A proc-block which takes a rank 4 tensor as input, whose dimension is of this form `[1, rows, columns, confidence]`.")
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("image")
        .with_tag("segmentation")
        .with_input(TensorMetadata::new("image")
        .with_description("An image-like tensor with the dimensions, `[1, rows, columns, category_confidence]`. Each \"pixel\" is associated with a set of confidence values, where each value indicates how confident the model is that the pixel is in that category."))
        .with_output(TensorMetadata::new("segmentation_map")
        .with_description(
"An image-like tensor where each pixel contains the index of the category with the highest confidence level."
        ))
        .with_output(
            TensorMetadata::new("indices")
                .with_description("The categories used in `segmentation_map`."),
        )
}

/// A proc-block which takes a rank 4 `tensor` as input, whose dimension is of
/// this form `[1, x, y, z]`.
///
/// It will return:
/// 1. a 2-d `tensor` after performing argmax along the axis-3 of the tensor
/// 2. a 1-d `tensor` which a `set` of all the number present in the above 2-d
///    `tensor`
struct SegmentOutput;

impl ProcBlock for SegmentOutput {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "input",
                ElementType::F32,
                vec![1, 0, 0, 0],
            )],
            outputs: vec![
                TensorConstraint::new(
                    "segmentation_map",
                    ElementType::U32,
                    vec![0, 0],
                ),
                TensorConstraint::new("indices", ElementType::U32, vec![0]),
            ],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let input = Tensor::get_named(&inputs, "input")?.view_4d::<f32>()?;

        let (segmented_map, indices) = transform(input);

        Ok(vec![
            Tensor::new("segmentation_map", &segmented_map),
            Tensor::new("indices", &indices),
        ])
    }
}

impl From<Vec<Argument>> for SegmentOutput {
    fn from(_: Vec<Argument>) -> Self { SegmentOutput }
}

fn transform(input: ArrayView4<'_, f32>) -> (Array2<u32>, Array1<u32>) {
    let (_, rows, columns, _) = input.dim();

    let mut map = Array2::zeros((rows, columns));
    let mut label_index = BTreeSet::new();

    for i in 0..rows {
        for j in 0..columns {
            let val = input.slice(s![0 as usize, i, j, ..]);
            let (index, _) =
                val.iter().enumerate().fold((0, 0.0), |max, (ind, &val)| {
                    if val > max.1 {
                        (ind, val)
                    } else {
                        max
                    }
                });
            map[[i, j]] = index as u32;
            label_index.insert(index as u32);
        }
    }

    (map, label_index.into_iter().collect())
}

#[cfg(test)]

mod tests {
    use hotg_rune_proc_blocks::ndarray::{self, Array3};

    use super::*;

    #[test]
    fn test_argmax() {
        let input: Array3<f32> = ndarray::array![
            [
                [1.7611206_f32, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
            [
                [1.7611206, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
            [
                [1.7611206, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
            [
                [1.7611206, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
            [
                [1.7611206, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
        ];
        let input = input.broadcast((1, 5, 4, 3)).unwrap();

        let (segments, indices) = transform(input);

        assert_eq!(indices, ndarray::array![1, 2]);
        let segments_should_be: Array2<u32> = ndarray::array![
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
        ];
        assert_eq!(segments, segments_should_be);
    }
}
