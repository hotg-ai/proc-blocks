#![cfg_attr(not(feature = "metadata"), no_std)]
use core::{
    fmt::Debug,
    ops::{Div, Sub},
};
use hotg_rune_proc_blocks::{Tensor, Transform};
use num_traits::ToPrimitive;

/// Normalize the input to the range `[0, 1]`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, hotg_rune_proc_blocks::ProcBlock,
)]
#[non_exhaustive]
#[transform(inputs = [f32; 1], outputs = [f32; 1])]
#[transform(inputs = [f32; 2], outputs = [f32; 2])]
#[transform(inputs = [f32; 3], outputs = [f32; 3])]
pub struct Normalize {}

impl<T> Transform<Tensor<T>> for Normalize
where
    T: ToPrimitive,
{
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<T>) -> Tensor<f32> {
        let (min, max) =
            min_max(input.elements().iter().map(|e| e.to_f32().unwrap()))
                .unwrap();
        let range =max-min;
        if range ==0.0 {
            return  Tensor::zeroed(input.dimensions().to_vec());
        }
        input.map(|_, element| {
            let element = element.to_f32().unwrap();
            (element - min) / range
        })
    }
}

fn min_max(items: impl Iterator<Item = f32>) -> Option<(f32, f32)> {
    items.into_iter().fold(None, |bounds, item| match bounds {
        Some((min, max)) => {
            let min = if item < min { item } else { min };
            let max = if max < item { item } else { max };
            Some((min, max))
        },
        None => Some((item, item)),
    })
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
                Metadata::new("Normalize", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Normalize a tensor's elements to the range, `[0, 1]`.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("normalize");

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

            let output = TensorMetadata::new("normalized");
            output.set_description("normalized tensor in the range [0, 1]");
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

    #[test]
    fn it_works() {
        let input = Tensor::from([0.0, 1.0, 2.0]);
        let mut pb = Normalize::default();

        let output = pb.transform(input);

        assert_eq!(output, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn it_works_with_integers() {
        let input: Tensor<i32> = Tensor::from([0, 1, 2]);
        let mut pb = Normalize::default();

        let output: Tensor<f32> = pb.transform(input);

        assert_eq!(output.elements(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn handle_empty() {
        let input = Tensor::from([0.0; 384]);
        let mut pb = Normalize::default();

        let output = pb.transform(input.clone());

        assert_eq!(output, input);
        assert_eq!(output.elements().len(), 384);
    }
}
