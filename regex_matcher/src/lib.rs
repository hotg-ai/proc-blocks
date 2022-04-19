#![cfg_attr(not(feature = "metadata"), no_std)]

#[macro_use]
extern crate alloc;

use alloc::{string::String};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct RegexMatcher {
    matching_regex: String
}

fn re_matchchar(regex_char: Option<char>, text_char: Option<char>) -> bool {
    regex_char == Some('.') && text_char.is_some() || regex_char == text_char
}

fn re_matchstar(c: char, regex: &str, r: usize, text: &str, mut t: usize) -> bool
{
    loop {
        // If the regex matches with the rest of the text, we have a  match
        if re_matchhere(regex, r, text, t) {
            return true;
        }

        // Otherwise continue consuming each character
        if !re_matchchar(Some(c), text.chars().nth(t)) {
            break;
        }

        t += 1;
    }

    false
}

// re_matchhere: search for regexp at beginning of text
fn re_matchhere(regex: &str, r: usize, text: &str, t: usize) -> bool {
    // The whole regex is consumed. We have a match
    if r >= regex.len() {
        return true;
    }

    // The main call that does backtracking to match a single *
    if regex.chars().nth(r + 1) == Some('*') {
        return re_matchstar(regex.chars().nth(r).unwrap(), regex, r + 2, text, t);
    }

    // For +, we can simply use re_matchstar, after making sure the first character matches
    if regex.chars().nth(r + 1) == Some('+') && re_matchchar(regex.chars().nth(r), text.chars().nth(t)) {
        return re_matchstar(regex.chars().nth(r).unwrap(), regex, r + 2, text, t + 1);
    }

    // Match end of the line
    if regex.chars().nth(r) == Some('$') && t == text.len() {
        return t == text.len();
    }

    // Match a single character
    // TODO: Add support for escape sequences
    if re_matchchar(regex.chars().nth(r), text.chars().nth(t)) {
        return re_matchhere(regex, r + 1, text, t + 1);
    }

    false
}

// re_match: search for regexp anywhere in text
// A super simple implementation based on: https://www.cs.princeton.edu/courses/archive/spr09/cos333/beautiful.html
// TODO: Support boolean operations
// TODO: Simply port all of this: https://github.com/kokke/tiny-regex-c/blob/master/re.c
fn re_match(regex: &str, r: usize, text: &str, mut t: usize) -> bool {
    if regex.starts_with('^') {
        return re_matchhere(regex, r + 1, text, t);
    }

    loop {
        if re_matchhere(regex,r, text, t) {
            return true;
        }

        t += 1;

        if t >= text.len() {
            break;
        }
    }

    false
}

impl Transform<Tensor<u8>> for RegexMatcher {
    type Output = Tensor<u8>;

    fn transform(
        &mut self,
        inputs: Tensor<u8>
    ) -> Tensor<u8> {
        let input_text = core::str::from_utf8(inputs.elements())
            .expect("Input tensor should be valid UTF8");

        // self.matching_regex ==
        Tensor::new_vector(vec![re_match(&self.matching_regex, 0, input_text, 0) as u8])
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
                Metadata::new("Regex Capturer", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Given a body of text and a regex, extract all parts of the text that match the regex.",
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

            let matching_regex = ArgumentMetadata::new("matching_regex");
            matching_regex.set_description("A basic regular expression that supports: ^, $, +, * operations");
            matching_regex.set_type_hint(TypeHint::OnelineString);
            metadata.add_argument(&matching_regex);

            let matched = TensorMetadata::new("matched");
            matched.set_description("A true/false tensor that specifies if the text matches the matching regex");
            let hint =
                supported_shapes(&[ElementType::Uint8], Dimensions::Fixed(&[0]));
            matched.add_hint(&hint);
            metadata.add_output(&matched);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]
    fn test_token_extractor() {

        fn str_tensor(s: &str) -> Tensor<u8> {
            Tensor::new_vector(s.as_bytes().to_vec())
        }

        let tensor_true = Tensor::new_vector(vec![1]);
        let tensor_false = Tensor::new_vector(vec![0]);

        let mut regex_matcher = RegexMatcher{ matching_regex: String::from("^abcd.*fg") };
        assert_eq!(regex_matcher.transform(str_tensor("abcdefg")), tensor_true);
        assert_eq!(regex_matcher.transform(str_tensor("abcdfg")), tensor_true);
        assert_eq!(regex_matcher.transform(str_tensor("asdf abcdfg")), tensor_false);
        assert_eq!(regex_matcher.transform(str_tensor("bcdefg")), tensor_false);
        assert_eq!(regex_matcher.transform(str_tensor("abefg")), tensor_false);

        let mut regex_matcher = RegexMatcher{ matching_regex: String::from("abcd.*fg$") };
        assert_eq!(regex_matcher.transform(str_tensor("abcdefg")), tensor_true);
        assert_eq!(regex_matcher.transform(str_tensor("abcdfg asdf")), tensor_false);

        let mut regex_matcher = RegexMatcher{ matching_regex: String::from("abcd.+fg") };
        assert_eq!(regex_matcher.transform(str_tensor("abcdefg")), tensor_true);
        assert_eq!(regex_matcher.transform(str_tensor("abcdfg")), tensor_false);
    }
}
