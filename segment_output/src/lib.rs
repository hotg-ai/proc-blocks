#![no_std]

extern crate alloc;
use alloc::sync::Arc;
use alloc::vec;
use alloc::{collections::btree_set::BTreeSet, vec::Vec};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc-block which takes a rank 4 `tensor` as input, whose dimension is of this form `[1, x, y, z]`. It will return:
/// 1) a 2-d `tensor` after performing argmax along the axis-3 of the tensor
/// 2) a 1-d `tensor` which a `set` of all the number present in the above 2-d `tensor`

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct SegmentOutput {}

impl SegmentOutput {
    pub const fn new() -> Self {
        SegmentOutput {}
    }
}

impl Default for SegmentOutput {
    fn default() -> Self {
        SegmentOutput::new()
    }
}

impl Transform<Tensor<f32>> for SegmentOutput {
    type Output = (Tensor<u32>, Tensor<u32>);

    fn transform(&mut self, input: Tensor<f32>) -> Self::Output {
        let dim = input.dimensions();
        match input.rank() {
            4 => {
                if dim[0] != 1 {
                    panic!("the first dimension should be 1")
                }
            }
            _ => panic!("it only accept a rank 4 tensor"),
        }
        let mut vec_2d: Vec<Vec<u32>> = Vec::new();
        let rows = dim[1] as usize;
        let columns = dim[2] as usize;

        let mut label_index = BTreeSet::new();

        for i in 0..rows {
            vec_2d.push(vec![]);
            for j in 0..columns {
                let val = input.slice::<1>(&[0, i, j]).unwrap();

                let (index, _) = val.elements().iter().enumerate().fold(
                    (0, 0.0),
                    |max, (ind, &val)| {
                        if val > max.1 {
                            (ind, val)
                        } else {
                            max
                        }
                    },
                ); // Doing argmax over the array
                vec_2d[i].push(index as u32);
                label_index.insert(index as u32);
            }
        }

        let elements: Arc<[u32]> = vec_2d
            .into_iter()
            .flat_map(|v: Vec<u32>| v.into_iter())
            .collect();
        (
            Tensor::new_row_major(elements, alloc::vec![rows, columns]),
            Tensor::new_vector(label_index),
        )
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]

    fn test_argmax() {
        let input = [[
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
            [
                [1.7611206, -0.824405, 3.3042068],
                [4.1308413, 3.8263698, 13.207806],
                [3.4352894, 4.6627636, 4.464175],
                [1.7611206, 8.24405, 3.3042068],
            ],
        ]];

        let input: Tensor<f32> = input.into();
        let mut argmax = SegmentOutput::default();
        let output = argmax.transform(input);
        let should_be = [
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
        ];
        let should_be: Tensor<u32> = should_be.into();
        let label_index: Tensor<u32> = [1, 2].into();
        assert_eq!(output, (should_be, label_index));
    }

    #[test]
    #[should_panic = "it only accept a rank 4 tensor"]
    fn not_rank_4_tensor() {
        let input = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]; // a rank 3 array
        let input: Tensor<f32> = input.into();
        let mut argmax = SegmentOutput::default();
        let _output = argmax.transform(input);
    }
    #[test]
    #[should_panic = "the first dimension should be 1"]
    fn first_dimension_not_1() {
        let input = [[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 6.0]]]]; // [3,1,1,2] dimension array
        let input: Tensor<f32> = input.into();
        let mut argmax = SegmentOutput::default();
        let _output = argmax.transform(input);
    }
}
