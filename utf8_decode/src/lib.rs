use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use std::borrow::Cow;

/// A proc block which can convert u8 bytes to utf8
#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct Utf8Decode {}

impl Transform<Tensor<u8>> for Utf8Decode {
    type Output = Tensor<Cow<'static, str>>;

    fn transform(&mut self, input: Tensor<u8>) -> Self::Output {
        let underlying_bytes: &[u8] = input.elements();

        let mut useful_bytes = &underlying_bytes[..underlying_bytes.len()];
        if let Some(index) = underlying_bytes.iter().position(|&x| x == 0) {
            useful_bytes = &underlying_bytes[..index];
        }

        let input_text = core::str::from_utf8(useful_bytes)
            .expect("Input tensor should be valid UTF8");

        let output_text = vec![Cow::Owned(input_text.to_string())];

        Tensor::new_vector(output_text)
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

            let metadata =
                Metadata::new("UTF8 Decode", env!("CARGO_PKG_VERSION"));
            metadata.set_description("Decode a string from UTF-8 bytes.");
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("text");
            metadata.add_tag("nlp");

            let input = TensorMetadata::new("bytes");
            input.set_description("The string as UTF-8 encoded bytes");
            let hint = supported_shapes(
                &[ElementType::Uint8],
                Dimensions::Fixed(&[0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("string");
            output.set_description("The decoded text.");
            let hint =
                supported_shapes(&[ElementType::Utf8], Dimensions::Fixed(&[1]));
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
    fn test_for_utf8_decoding() {
        let mut cast = Utf8Decode::default();
        let bytes: Vec<u8> = "Hi, use me to convert your u8 bytes to utf8."
            .as_bytes()
            .to_vec();
        let input = Tensor::new_vector(bytes);

        let output = cast.transform(input);
        let should_be = Tensor::new_vector(vec![Cow::Borrowed(
            "Hi, use me to convert your u8 bytes to utf8.",
        )]);

        assert_eq!(output, should_be);
    }
}
