use crate::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError,
    },
    runtime_v1::{
        ArgumentMetadata, ArgumentType, Dimensions, ElementType, GraphContext,
        KernelContext, Metadata, TensorMetadata, TensorParam, TensorResult,
    },
};
use hotg_rune_proc_blocks::{ndarray::ArrayViewD, BufferExt};
use line_span::LineSpans;
use std::{fmt::Debug, ops::Range};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Label", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
                "Using a wordlist, retrieve the label that corresponds to each element in a tensor.",
            );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("classify");

        let labels = ArgumentMetadata::new("wordlist");
        let hint =
            runtime_v1::supported_argument_type(ArgumentType::LongString);
        labels.add_hint(&hint);
        metadata.add_argument(&labels);

        let fallback = ArgumentMetadata::new("fallback");
        fallback.set_default_value("");
        fallback
            .set_description("The label to use if an index is out of bounds");
        let hint = runtime_v1::supported_argument_type(ArgumentType::String);
        fallback.add_hint(&hint);
        metadata.add_argument(&fallback);

        let indices = TensorMetadata::new("indices");
        indices.set_description("Indices for labels in the wordlist.");
        let hint = runtime_v1::supported_shapes(
            &[ElementType::U32],
            Dimensions::Dynamic,
        );
        indices.add_hint(&hint);
        metadata.add_input(&indices);

        let output = TensorMetadata::new("labels");
        output.set_description("The corresponding labels.");
        let hint = runtime_v1::supported_shapes(
            &[ElementType::Utf8],
            Dimensions::Dynamic,
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        runtime_v1::register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or_else(|| GraphError::MissingContext)?;

        let _ = get_wordlist(|n| ctx.get_argument(n))
            .map_err(GraphError::InvalidArgument)?;

        ctx.add_input_tensor("indices", ElementType::U32, Dimensions::Dynamic);
        ctx.add_output_tensor("labels", ElementType::Utf8, Dimensions::Dynamic);

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or_else(|| KernelError::MissingContext)?;

        let wordlist = get_wordlist(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let TensorResult {
            buffer,
            dimensions,
            element_type,
        } = ctx.get_input_tensor("indices").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "indices".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let indices = match element_type {
            ElementType::U32 => {
                buffer.view::<u32>(&dimensions).map_err(|e| {
                    KernelError::InvalidInput(InvalidInput {
                        name: "indices".to_string(),
                        reason: BadInputReason::InvalidValue(e.to_string()),
                    })
                })?
            },
            _ => todo!(),
        };

        let fallback = ctx.get_argument("fallback").unwrap_or_default();
        let serialized_labels = label(indices, &wordlist, &fallback);

        ctx.set_output_tensor(
            "labels",
            TensorParam {
                element_type: ElementType::Utf8,
                dimensions: &dimensions,
                buffer: &serialized_labels,
            },
        );

        Ok(())
    }
}

fn label(
    indices: ArrayViewD<'_, u32>,
    wordlist: &Lines,
    fallback: &str,
) -> Vec<u8> {
    let labels = indices.map(|&index| match wordlist.get(index as usize) {
        Some(label) => label,
        None => fallback,
    });

    hotg_rune_proc_blocks::string_tensor_from_ndarray(&labels)
}

fn get_wordlist(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<Lines, InvalidArgument> {
    let wordlist = get_argument("wordlist").ok_or_else(|| InvalidArgument {
        name: "wordlist".to_string(),
        reason: BadArgumentReason::NotFound,
    })?;

    Ok(Lines::new(wordlist))
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

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

    #[test]
    fn get_the_correct_labels() {
        let wordlist = "zero\none\ntwo\nthree";
        let wordlist = Lines::new(wordlist.to_string());
        let indices = ndarray::aview1(&[2_u32, 0, 1]);

        let serialized = label(indices.into_dyn(), &wordlist, "...");

        let expected = ndarray::arr1(&["two", "zero", "one"]).into_dyn();
        let got = serialized.string_view(&[3]).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn label_index_out_of_bounds_uses_fallback() {
        let wordlist = "zero\none\ntwo\nthree";
        let wordlist = Lines::new(wordlist.to_string());
        let indices = ndarray::aview1(&[100_u32]);

        let serialized = label(indices.into_dyn(), &wordlist, "UNKNOWN");

        let expected = ndarray::arr1(&["UNKNOWN"]).into_dyn();
        let got = serialized.string_view(&[1]).unwrap();
        assert_eq!(got, expected);
    }
}
