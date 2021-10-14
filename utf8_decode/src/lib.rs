#![no_std]

extern crate alloc;

use crate::alloc::string::ToString;
use alloc::borrow::Cow;
use alloc::vec;
use core::str;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct Utf8Decode {}

impl Transform<Tensor<u8>> for Utf8Decode {
    type Output = Tensor<Cow<'static, str>>;

    fn transform(&mut self, input: Tensor<u8>) -> Self::Output {
        let underlying_bytes: &[u8] = input.elements();

        let mut index = underlying_bytes.len();
        for (ind, value) in underlying_bytes.iter().enumerate() {
            if value == &0_u8 {
                index = ind
            }
        }
        let underlying_bytes = &underlying_bytes[..index];

        let input_text = core::str::from_utf8(underlying_bytes)
            .expect("Input tensor should be valid UTF8");

        let output_text = vec![Cow::Owned(input_text.to_string())];

        Tensor::new_vector(output_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    #[test]

    fn test_for_f32() {
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
