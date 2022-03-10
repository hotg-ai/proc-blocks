#![cfg_attr(not(feature = "metadata"), no_std)]

extern crate alloc;

use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use num_traits::{Bounded, ToPrimitive};

/// A normalization routine which takes some tensor of integers and fits their
/// values to the range `[0, 1]` as `f32`'s.
#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
#[non_exhaustive]
#[transform(inputs = [u8; _], outputs = [f32; _])]
#[transform(inputs = [i8; _], outputs = [f32; _])]
#[transform(inputs = [u16; _], outputs = [f32; _])]
#[transform(inputs = [i16; _], outputs = [f32; _])]
#[transform(inputs = [u32; _], outputs = [f32; _])]
#[transform(inputs = [i32; _], outputs = [f32; _])]
pub struct ImageNormalization {}

impl ImageNormalization {
    fn check_input_dimensions(&self, dimensions: &[usize]) {
        match *dimensions {
            [_, _, _, 3] => {},
            [_, _, _, channels] => panic!(
                "The number of channels should be either 1 or 3, found {}",
                channels
            ),
            _ => panic!("The image normalization proc block only supports outputs of the form [frames, rows, columns, channels], found {:?}", dimensions),
        }
    }
}

impl<T> Transform<Tensor<T>> for ImageNormalization
where
    T: Bounded + ToPrimitive + Copy,
{
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        self.check_input_dimensions(input.dimensions());
        input.map(|_, &value| normalize(value).expect("Cast should never fail"))
    }
}

fn normalize<T>(value: T) -> Option<f32>
where
    T: Bounded + ToPrimitive,
{
    let min = T::min_value().to_f32()?;
    let max = T::max_value().to_f32()?;
    let value = value.to_f32()?;
    debug_assert!(min <= value && value <= max);

    Some((value - min) / (max - min))
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

            let metadata =
                Metadata::new("Image Normalization", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Normalize the pixels in an image to the range `[0, 1]`",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("image");
            metadata.add_tag("normalize");

            let input = TensorMetadata::new("image");
            input.set_description("An image with the dimensions `[1, width, height, channels]`.\n\nRGB images typically have 3 channels and grayscale images have 1.");
            let hint = supported_shapes(
                &[
                    ElementType::Uint8,
                    ElementType::Int8,
                    ElementType::Uint16,
                    ElementType::Int16,
                    ElementType::Uint32,
                    ElementType::Int32,
                ],
                Dimensions::Fixed(&[1, 0, 0, 0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("normalized");
            output.set_description(
                "The image's pixels, normalized to the range `[0, 1]`.",
            );
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[1, 0, 0, 0]),
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
    use alloc::vec;

    #[test]
    fn normalizing_with_default_distribution_is_noop() {
        let dims = vec![1, 1, 1, 3];
        let input: Tensor<u8> =
            Tensor::new_row_major(vec![0, 127, 255].into(), dims.clone());
        let mut norm = ImageNormalization::default();
        let should_be: Tensor<f32> =
            Tensor::new_row_major(vec![0.0, 127.0 / 255.0, 1.0].into(), dims);

        let got = norm.transform(input);

        assert_eq!(got, should_be);
    }
}
