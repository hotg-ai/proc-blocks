#![cfg_attr(not(feature = "metadata"), no_std)]
use core::ops::{Add, Div};

use hotg_rune_core::AsElementType;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use ndarray::ArrayViewD;
use num_traits::{ToPrimitive, Zero};

#[derive(Debug, Default, Clone, Copy, PartialEq, ProcBlock)]
pub struct Mean {}

impl Mean {
    pub fn new() -> Self { Mean {} }
}

impl<'a, T> Transform<Tensor<T>> for Mean
where
    T: Clone
        + ToPrimitive
        + Add<Output = T>
        + Div<Output = T>
        + Zero
        + AsElementType,
{
    // TODO: Figure out whether the user will *always* want floats out, or
    // whether the output type should match the input.
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<T>) -> Tensor<f32> {
        let tensor = ArrayViewD::from_shape(
            input.shape().dimensions(),
            input.elements(),
        )
        .expect("Unable to get a tensor view");

        let mean = tensor.map(|v| v.to_f32().unwrap()).std(1.0);

        Tensor::single(mean)
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

            let metadata =
                Metadata::new("Standard Deviation", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Calculate the standard deviation of all the elements in a tensor",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));

            let input = TensorMetadata::new("input");
            let supported_types = [
                ElementType::Uint8,
                ElementType::Int8,
                ElementType::Uint16,
                ElementType::Int16,
                ElementType::Uint32,
                ElementType::Int32,
                ElementType::Float32,
                ElementType::Uint64,
                ElementType::Int64,
                ElementType::Float64,
            ];
            let hint = supported_shapes(&supported_types, Dimensions::Dynamic);
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("std_dev");
            let hint =
                supported_shapes(&supported_types, Dimensions::Fixed(&[1]));
            output.add_hint(&hint);
            metadata.add_output(&output);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn mean_of_1d_tensor() {
        let mut m = Mean::new();
        let input = Tensor::new_vector(alloc::vec![1_u32, 2, 3, 4, 5, 6]);

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(1.8708287));
    }

    #[test]
    fn mean_of_multidimensional_tensor() {
        let mut m = Mean::new();
        let input = Tensor::new_row_major(
            alloc::vec![1_u32, 2, 3, 4, 5, 6].into(),
            alloc::vec![2, 3],
        );

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(1.8708287));
    }
}
