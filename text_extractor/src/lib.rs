use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementType, InvalidInput, Metadata, ProcBlock, RunError,
        Tensor, TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: TextExtractor,
}

fn metadata() -> Metadata {
    Metadata::new("Text Extractor", env!("CARGO_PKG_VERSION"))
        .with_description(
                "Given a body of text and some start/end indices, extract parts of the text (i.e. words/phrases) specified by those indices.",
            )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("nlp")
        .with_input(
            TensorMetadata::new("text")
                .with_description("The tokens making up this body of text."),
        )
        .with_input(
            TensorMetadata::new("start_logits")
                .with_description("The indices for the start of each word/phrase to extract."),
        )
        .with_input(
            TensorMetadata::new("end_logits")
                .with_description("The indices for the end of each word/phrase to extract."),
        )
        .with_output(
            TensorMetadata::new("phrases")
                .with_description("The phrases that were extracted.")
        )
}

#[derive(Debug, Default, Clone, PartialEq)]
struct TextExtractor;

impl ProcBlock for TextExtractor {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new("wordlist", ElementType::Utf8, [0]),
                TensorConstraint::new("start_logits", ElementType::U32, [0]),
                TensorConstraint::new("end_logits", ElementType::U32, [0]),
            ],
            outputs: vec![TensorConstraint::new(
                "phrases",
                ElementType::Utf8,
                [0],
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let text = Tensor::get_named(&inputs, "text")?.string_view()?;
        let start_logits =
            Tensor::get_named(&inputs, "start_logits")?.view_1d::<u32>()?;
        let end_logits =
            Tensor::get_named(&inputs, "end_logits")?.view_1d::<u32>()?;

        let mut phrases = Vec::new();

        for (i, (&start, &end)) in
            start_logits.into_iter().zip(end_logits).enumerate()
        {
            let start = start as usize;
            let end = end as usize;

            if start == 0 && end == 0 {
                // No more logits
                break;
            } else if start > end {
                return Err(RunError::other(format!("At index {i}, the start logit ({start}) is after the end ({end})")));
            } else if end > text.len() {
                return Err(InvalidInput::invalid_value("end_logits", format!("The {i}'th logit, {end}, is out of bounds (num tokens: {})", text.len())).into());
            } else if start >= text.len() {
                return Err(InvalidInput::invalid_value("start_logits", format!("The {i}'th logit, {start}, is out of bounds (num tokens: {})", text.len())).into());
            }

            dbg!(start, end);
            let tokens = text.slice(ndarray::s!(start..=end));
            let phrase = merge_phrases(tokens.iter().copied());
            phrases.push(phrase);
        }

        let phrases = ndarray::aview1(&phrases);
        Ok(vec![Tensor::from_strings("phrases", &phrases)])
    }
}

impl From<Vec<Argument>> for TextExtractor {
    fn from(_: Vec<Argument>) -> Self { TextExtractor::default() }
}

fn merge_phrases<'a>(tokens: impl Iterator<Item = &'a str>) -> String {
    let mut buffer = String::new();

    for token in tokens {
        match token.strip_prefix("##") {
            Some(token) => buffer.push_str(token),
            None => {
                if !buffer.is_empty() {
                    buffer.push_str(" ");
                }
                buffer.push_str(token);
            },
        }
    }

    buffer
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn known_inputs() {
        let proc_block = TextExtractor::default();
        let words = ndarray::array![
            "[UNK]", "[UNK]", "una", "##ffa", "##ble", "world", "!"
        ];
        let inputs = vec![
            Tensor::from_strings("text", &words),
            Tensor::new_1d("start_logits", &[2_u32]),
            Tensor::new_1d("end_logits", &[4_u32]),
        ];

        let got = proc_block.run(inputs).unwrap();

        let should_be = vec![Tensor::from_strings(
            "phrases",
            &ndarray::array!["unaffable"],
        )];
        assert_eq!(got, should_be);
    }
}
