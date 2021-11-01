#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct TextExtractor {}

impl TextExtractor {
    pub const fn new() -> Self {
        TextExtractor {}
    }
}

impl Default for TextExtractor {
    fn default() -> Self {
        TextExtractor::new()
    }
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
