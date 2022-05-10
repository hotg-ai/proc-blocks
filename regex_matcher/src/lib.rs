use crate::{
    proc_block_v1::{GraphError, KernelError},
    runtime_v1::{
        supported_shapes, Dimensions, ElementType, GraphContext, KernelContext,
        Metadata, TensorMetadata, TensorParam, TensorResult,
    },
};
use hotg_rune_proc_blocks::{BufferExt};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

fn re_matchchar(regex_char: Option<char>, text_char: Option<char>) -> bool {
    regex_char == Some('.') && text_char.is_some() || regex_char == text_char
}

fn re_matchstar(
    c: char,
    regex: &str,
    r: usize,
    text: &str,
    mut t: usize,
) -> bool {
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
        return re_matchstar(
            regex.chars().nth(r).unwrap(),
            regex,
            r + 2,
            text,
            t,
        );
    }

    // For +, we can simply use re_matchstar, after making sure the first character matches
    if regex.chars().nth(r + 1) == Some('+')
        && re_matchchar(regex.chars().nth(r), text.chars().nth(t))
    {
        return re_matchstar(
            regex.chars().nth(r).unwrap(),
            regex,
            r + 2,
            text,
            t + 1,
        );
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
        if re_matchhere(regex, r, text, t) {
            return true;
        }

        t += 1;

        if t >= text.len() {
            break;
        }
    }

    false
}

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Regex Matched", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("string");
        metadata.add_tag("nlp");
        let text_sample = TensorMetadata::new("text");
        text_sample.set_description("Text to run regex on");
        let hint = supported_shapes(&[ElementType::Utf8], Dimensions::Dynamic);
        text_sample.add_hint(&hint);
        metadata.add_input(&text_sample);

        let matching_regex = TensorMetadata::new("matching_regex");
        matching_regex.set_description(
            "A basic regular expression that supports: ^, $, +, * operations",
        );
        let hint = supported_shapes(&[ElementType::Utf8], Dimensions::Dynamic);
        matching_regex.add_hint(&hint);
        metadata.add_input(&matching_regex);

        let matched = TensorMetadata::new("match");
        matched.set_description("A true/false tensor that specifies if the text matches the matching regex");
        let hint = supported_shapes(&[ElementType::U8], Dimensions::Dynamic);
        matched.add_hint(&hint);
        metadata.add_input(&matched);

        runtime_v1::register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&id).unwrap();

        ctx.add_input_tensor(
            "text",
            ElementType::Utf8,
            Dimensions::Fixed(&[0]),
        );
        ctx.add_output_tensor(
            "matching_regex",
            ElementType::Utf8,
            Dimensions::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).unwrap();

        let TensorResult {
            element_type,
            buffer,
            dimensions: _,
        } = ctx.get_input_tensor("text").unwrap();

        let text = match element_type {
            ElementType::Utf8 => buffer
                .strings()
                .map_err(|e| KernelError::Other(e.to_string()))?,
            other => {
                return Err(KernelError::Other(format!(
                "The proc-block only accepts Utf8 tensors, found for text {:?}",
                other,
                )))
            },
        };

        let TensorResult {
            element_type,
            buffer,
            dimensions: _,
        } = ctx.get_input_tensor("matching_regex").unwrap();

        let regex = match element_type {
            ElementType::Utf8 => buffer
                .strings()
                .map_err(|e| KernelError::Other(e.to_string()))?,
            other => {
                return Err(KernelError::Other(format!(
                "The regex proc-block only accepts Utf8 tensors, found for regex {:?}",
                other,
                )))
            },
        };


        ctx.set_output_tensor(
            "matched",
            TensorParam {
                element_type: ElementType::Utf8,
                dimensions: &[0],
                buffer: &vec![
                    re_match(&regex[0][..], 0, text[0], 0) as u8
                ][..],
            },
        );

        Ok(())
    }
}
