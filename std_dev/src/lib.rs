#![cfg_attr(not(feature = "metadata"), no_std)]

use hotg_rune_core::AsElementType;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use ndarray::ArrayViewD;
use num_traits::{Float, FromPrimitive};
#[derive(Debug, Default, Clone, Copy, PartialEq, ProcBlock)]
pub struct StdDev {}

impl StdDev {
    pub fn new() -> Self { StdDev {} }
}

impl<'a, T> Transform<Tensor<T>> for StdDev
where
    T: Float + AsElementType + FromPrimitive
{
    // TODO: Figure out whether the user will *always* want floats out, or
    // whether the output type should match the input.
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<T>) -> Tensor<T> {
        let tensor = ArrayViewD::from_shape(
            input.shape().dimensions(),
            input.elements(),
        )
        .expect("Unable to get a tensor view");
        let mean = tensor.mean().unwrap_or_else(T::one);
        let mut sum_sq = T::zero();
        tensor.for_each(|&t| {
            sum_sq = sum_sq + (t - mean).powi(2);
        });
        Tensor::single((sum_sq / T::from_usize(tensor.len()).unwrap()).sqrt())
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
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[1]),
            );
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
    fn stddev_of_1d_tensor() {
        let mut m = StdDev::new();
        let input =
            Tensor::new_vector(alloc::vec![70.0_f32, 96.0, 58.0, 74.0, 38.0, 69.0, 76.0, 44.0, 23.0, 82.0]);

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(21.061813));
    }

    #[test]
    fn stddev_of_multidimensional_tensor() {
        let mut m = StdDev::new();
        let input = Tensor::new_row_major(
            alloc::vec![70.0_f32, 96.0, 58.0, 74.0, 38.0, 0.0].into(),
            alloc::vec![2, 3],
        );

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(30.50683));
    }
}
