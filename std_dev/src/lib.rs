#![cfg_attr(not(feature = "metadata"), no_std)]

use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use num_traits::{Float, FromPrimitive, ToPrimitive};

#[derive(Debug, Default, Clone, Copy, PartialEq, ProcBlock)]
pub struct StdDev {}

impl StdDev {
    pub fn new() -> Self { StdDev {} }
}

impl<'a, T> Transform<Tensor<T>> for StdDev
where
    T: Clone + ToPrimitive + FromPrimitive + Float,
{
    // TODO: Figure out whether the user will *always* want floats out, or
    // whether the output type should match the input.
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<T>) -> Tensor<T> {
        let mut mean = T::zero();
        let mut sum_sq = T::zero();
        let mut i = 0;
        input.elements().iter().for_each(|&x| {
            let count = T::from_usize(i + 1)
                .expect("Converting index to `T` must not fail.");
            let delta = x - mean;
            mean = mean + delta / count;
            sum_sq = (x - mean).mul_add(delta, sum_sq);
            i += 1;
        });

        Tensor::single(mean.sqrt())
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
    fn mean_of_1d_tensor() {
        let mut m = StdDev::new();
        let input =
            Tensor::new_vector(alloc::vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(1.8708287));
    }

    #[test]
    fn mean_of_multidimensional_tensor() {
        let mut m = StdDev::new();
        let input = Tensor::new_row_major(
            alloc::vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into(),
            alloc::vec![2, 3],
        );

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(1.8708287));
    }
}
