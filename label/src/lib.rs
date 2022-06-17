use hotg_rune_proc_blocks::guest::{
    parse, Argument, ArgumentMetadata, ArgumentType, CreateError, Dimensions,
    ElementType, Metadata, ProcBlock, RunError, Tensor, TensorConstraint,
    TensorConstraints, TensorMetadata,
};
use line_span::LineSpans;
use std::{fmt::Debug, ops::Range, str::FromStr};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Labels,
}

fn metadata() -> Metadata {
    Metadata::new("Label", env!("CARGO_PKG_VERSION"))
        .with_description(
                "Using a wordlist, retrieve the label that corresponds to each element in a tensor.",
            )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("classify")
        .with_argument(ArgumentMetadata::new("wordlist")
        .with_hint(ArgumentType::LongString)
    )
        .with_argument(ArgumentMetadata::new("fallback")
        .with_hint(ArgumentType::String)
        .with_default_value("")
    )
    .with_input(TensorMetadata::new("indices").with_description("Indices for labels in the wordlist."))
    .with_output(TensorMetadata::new("labels").with_description("The corresponding labels."))
}

#[derive(Debug, Clone, PartialEq)]
struct Labels {
    fallback: String,
    wordlist: Lines,
}

impl ProcBlock for Labels {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "indices",
                ElementType::U32,
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::new(
                "labels",
                ElementType::Utf8,
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let indices = Tensor::get_named(&inputs, "indices")?.view::<u32>()?;

        let labels = indices.mapv(|ix| {
            self.wordlist
                .get(ix as usize)
                .unwrap_or(self.fallback.as_str())
        });

        Ok(vec![Tensor::from_strings("labels", &labels)])
    }
}

impl TryFrom<Vec<Argument>> for Labels {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let wordlist = parse::required_arg(&args, "wordlist")?;
        let fallback =
            parse::optional_arg(&args, "fallback")?.unwrap_or_default();

        Ok(Labels { wordlist, fallback })
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Lines {
    text: String,
    lines: Vec<Range<usize>>,
}

impl Lines {
    fn new(text: String) -> Self {
        let lines = text.line_spans().map(|s| s.range()).collect();

        Lines { text, lines }
    }

    fn get(&self, line_number: usize) -> Option<&str> {
        let span = self.lines.get(line_number)?.clone();
        Some(&self.text[span])
    }
}

impl FromStr for Lines {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Lines::new(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

    #[test]
    fn get_the_correct_labels() {
        let wordlist = "zero\none\ntwo\nthree";
        let wordlist = Lines::new(wordlist.to_string());
        let indices = ndarray::aview1(&[2_u32, 0, 1]);
        let proc_block = Labels {
            wordlist,
            fallback: "...".to_string(),
        };

        let got = proc_block
            .run(vec![Tensor::new("indices", &indices)])
            .unwrap();

        let should_be = vec![Tensor::from_strings(
            "labels",
            &ndarray::arr1(&["two", "zero", "one"]),
        )];
        assert_eq!(got, should_be);
    }

    #[test]
    fn label_index_out_of_bounds_uses_fallback() {
        let wordlist = "zero\none\ntwo\nthree";
        let wordlist = Lines::new(wordlist.to_string());
        let indices = ndarray::aview1(&[100_u32]);
        let proc_block = Labels {
            wordlist,
            fallback: "...".to_string(),
        };

        let got = proc_block
            .run(vec![Tensor::new("indices", &indices)])
            .unwrap();

        let should_be =
            vec![Tensor::from_strings("labels", &ndarray::arr1(&["..."]))];
        assert_eq!(got, should_be);
    }
}
