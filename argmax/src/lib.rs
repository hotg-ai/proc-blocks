#![cfg_attr(not(feature = "metadata"), no_std)]
#[macro_use]
extern crate alloc;

use alloc::{sync::Arc, vec::Vec};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
// use core::sync::Arc;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Argmax {}

impl Argmax {
    pub const fn new() -> Self {
        Argmax {}
    }
}

impl Default for Argmax {
    fn default() -> Self {
        Argmax::new()
    }
}

impl Transform<Tensor<f32>> for Argmax {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        let dim = input.dimensions(); //[1, 26, 37]
        let input = input.elements();
        let len = dim.len();

        if len == 1 {
            let (index, _) =
                input.iter().enumerate().fold((0, 0.0), |max, (ind, &val)| {
                    if val > max.1 {
                        (ind, val)
                    } else {
                        max
                    }
                });

            let v: Vec<u32> = vec![index as u32];

            return Tensor::new_vector(v);
        } else {
            let mut vec_1d: Vec<u32> = Vec::new();
            let j = dim[2];

            for i in 0..dim[1] {
                let (ind, _) =
                    input[i * j..(i + 1) * j].iter().enumerate().fold(
                        (0, 0.0),
                        |max, (ind, &val)| {
                            if val > max.1 {
                                (ind, val)
                            } else {
                                max
                            }
                        },
                    );
                vec_1d.push(ind as u32);
            }
            let elements: Arc<[u32]> = vec_1d.into();
            Tensor::new_row_major(elements, vec![1, dim[1]])
        }
    }
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
    wit_bindgen_rust::export!("../wit-files/rune/rune-v1.wit");

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata = Metadata::new("Arg Max", env!("CARGO_PKG_VERSION"));
            metadata.set_description("Find the index of the largest element.");
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("max");
            metadata.add_tag("index");
            metadata.add_tag("numeric");

            let input = TensorMetadata::new("input");
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let max = TensorMetadata::new("max_index");
            max.set_description(
                "The index of the element with the highest value",
            );
            let hint = supported_shapes(
                &[ElementType::Uint32],
                Dimensions::Fixed(&[1]),
            );
            max.add_hint(&hint);
            metadata.add_output(&max);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_argmax() {
        let v = Tensor::new_vector(vec![2.3, 12.4, 55.1, 15.4]);
        let mut argmax = Argmax::default();
        let output = argmax.transform(v);
        let should_be: Tensor<u32> = [2].into();
        assert_eq!(output, should_be);
    }
    #[test]
    fn test_multi_dimension() {
        let v = [[
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]]
        .into();
        let mut argmax = Argmax::default();
        let output = argmax.transform(v);
        let should_be: Tensor<u32> = [[3, 3, 3]].into();
        assert_eq!(output, should_be);
    }
}
