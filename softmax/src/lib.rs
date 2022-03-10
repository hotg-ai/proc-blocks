#![cfg_attr(not(feature = "metadata"), no_std)]

extern crate alloc;

use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use libm::expf;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Softmax {}

impl Softmax {
    pub const fn new() -> Self { Softmax {} }
}

impl Default for Softmax {
    fn default() -> Self { Softmax::new() }
}

impl Transform<Tensor<f32>> for Softmax {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let b = input.map(|_, &x| expf(x as f32));
        let sum: f32 = b.elements().iter().sum();

        b.map(|_, &x| x / sum)
    }
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!(
        "../wit-files/rune/runtime-v1.wit"
    );
    wit_bindgen_rust::export!(
        "../wit-files/rune/rune-v1.wit"
    );

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata = Metadata::new("Softmax", env!("CARGO_PKG_VERSION"));
            metadata.set_description("Apply [softmax](https://en.wikipedia.org/wiki/Softmax_function), sometimes referred to as the *\"normalized exponential function\"* to each element in the tensor.");
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));

            let input = TensorMetadata::new("input");
            let hint =
                supported_shapes(&[ElementType::Float32], Dimensions::Dynamic);
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("output");
            output.set_description("The result of `exp(x)` to each element, `x`, in the tensor and dividing by the sum.");
            let hint =
                supported_shapes(&[ElementType::Float32], Dimensions::Dynamic);
            output.add_hint(&hint);
            metadata.add_output(&output);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_softmax() {
        let v = Tensor::new_vector(vec![2.3, 12.4, 5.1]);
        let mut softmax = Softmax::default();
        let output = softmax.transform(v);
        let should_be =
            Tensor::new_vector(vec![0.000041050153, 0.99928397, 0.00067505526]);
        assert_eq!(output, should_be);
    }
}
