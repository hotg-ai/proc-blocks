#![cfg_attr(not(feature = "metadata"), no_std)]

#[macro_use]
extern crate alloc;

use alloc::{borrow::Cow, string::String, vec::Vec};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct TextExtractor {}

impl TextExtractor {
    pub const fn new() -> Self { TextExtractor {} }
}

impl Default for TextExtractor {
    fn default() -> Self { TextExtractor::new() }
}

impl Transform<(Tensor<u8>, Tensor<u32>, Tensor<u32>)> for TextExtractor {
    type Output = Tensor<Cow<'static, str>>;

    fn transform(
        &mut self,
        inputs: (Tensor<u8>, Tensor<u32>, Tensor<u32>),
    ) -> Tensor<Cow<'static, str>> {
        let (text, start_logits, end_logits) = inputs;

        let underlying_bytes: &[u8] = text.elements();
        let input_text = core::str::from_utf8(underlying_bytes)
            .expect("Input tensor should be valid UTF8");

        let input_text: Vec<&str> = input_text.lines().collect();

        let start_index: u32 = start_logits.elements()[0];
        let end_index: u32 = end_logits.elements()[0];
        if end_index <= start_index {
            panic!(
                "Start index: {} is greater than or equal to end index: {}",
                start_index, end_index
            );
        }

        let v = &input_text[start_index as usize..end_index as usize + 1];

        let mut buffer = String::new();
        for tok in v {
            if let Some(s) = tok.strip_prefix("##") {
                buffer.push_str(s);
            } else {
                if !buffer.is_empty() {
                    buffer.push_str(" ");
                }
                buffer.push_str(tok);
            }
        }

        let output_text = vec![Cow::Owned(buffer)];

        Tensor::new_vector(output_text)
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

            let metadata =
                Metadata::new("Text Extractor", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Given a body of text and some start/end indices, extract parts of the text (i.e. words/phrases) specified by those indices.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("nlp");

            let text = TensorMetadata::new("text");
            text.set_description("A string of text.");
            let hint =
                supported_shapes(&[ElementType::Uint8], Dimensions::Fixed(&[0]));
            text.add_hint(&hint);
            metadata.add_input(&text);

            let start_logits = TensorMetadata::new("start_logits");
            start_logits.set_description(
                "The indices for the start of each word/phrase to extract.",
            );
            let hint = supported_shapes(
                &[ElementType::Uint32],
                Dimensions::Fixed(&[0]),
            );
            start_logits.add_hint(&hint);
            metadata.add_input(&start_logits);

            let end_logits = TensorMetadata::new("end_logits");
            end_logits.set_description(
                "The indices for the end of each word/phrase to extract.",
            );
            let hint = supported_shapes(
                &[ElementType::Uint32],
                Dimensions::Fixed(&[0]),
            );
            end_logits.add_hint(&hint);
            metadata.add_input(&end_logits);

            let phrases = TensorMetadata::new("phrases");
            phrases.set_description("The phrases that were extracted.");
            let hint =
                supported_shapes(&[ElementType::Utf8], Dimensions::Fixed(&[0]));
            phrases.add_hint(&hint);
            metadata.add_output(&phrases);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]
    fn test_token_extractor() {
        let bytes: Vec<u8> = "[UNK]\n[UNK]\nuna\n##ffa\n##ble\nworld\n!"
            .as_bytes()
            .to_vec();
        let bytes = Tensor::new_vector(bytes);
        let start_index = Tensor::new_vector(vec![2]);
        let end_index = Tensor::new_vector(vec![4]);

        let mut text_extractor = TextExtractor::default();
        let output = text_extractor.transform((bytes, start_index, end_index));

        let should_be = Tensor::new_vector(vec![Cow::Borrowed("unaffable")]);

        assert_eq!(output, should_be);
    }
}
