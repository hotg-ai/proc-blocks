#![cfg_attr(not(feature = "metadata"), no_std)]

#[macro_use]
extern crate alloc;

use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc-block which takes a rank 1 `tensor` as input, return 1 if value
/// inside the tensor is greater than 1 otherwise 0.
#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct BinaryClassification {
    threshold: f32,
}

impl BinaryClassification {
    pub const fn new() -> Self { BinaryClassification { threshold: 0.5 } }
}

impl Default for BinaryClassification {
    fn default() -> Self { BinaryClassification::new() }
}

impl Transform<Tensor<f32>> for BinaryClassification {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        // let value = input.into();
        let mut label: u32 = 0;
        if input.elements() > &[self.threshold] {
            label = 1
        }
        let v: Vec<u32> = vec![label];

        Tensor::new_vector(v)
    }
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!(
        "$CARGO_MANIFEST_DIR/../wit-files/rune/runtime-v1.wit"
    );
    wit_bindgen_rust::export!(
        "$CARGO_MANIFEST_DIR/../wit-files/rune/rune-v1.wit"
    );

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata = Metadata::new(
                "Binary Classification",
                env!("CARGO_PKG_VERSION"),
            );
            metadata.set_description(
                "Classify each element in a tensor depending on whether they are above or below a certain threshold.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("classify");

            let threshold = ArgumentMetadata::new("threshold");
            threshold.set_default_value("0.5");
            threshold.set_description("The classification threshold.");
            threshold.set_type_hint(TypeHint::Float);
            metadata.add_argument(&threshold);

            let input = TensorMetadata::new("input");
            input.set_description("The numbers to classify");
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("classified");
            output.set_description("A tensor of `1`'s and `0`'s, where `1` indicates an element was above the `threshold` and `0` means it was below.");
            let hint = supported_shapes(
                &[ElementType::Uint32],
                Dimensions::Fixed(&[0]),
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

    #[test]
    fn test_binary_classification() {
        let v = Tensor::new_vector(vec![0.7]);
        let mut bin_class = BinaryClassification::default();
        let output = bin_class.transform(v);
        let should_be: Tensor<u32> = [1].into();
        assert_eq!(output, should_be);
    }
}
