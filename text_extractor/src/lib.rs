use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{
    ndarray, runtime_v1::*, string_tensor_from_ndarray, BufferExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

#[macro_use]
extern crate alloc;
use alloc::{string::String, vec::Vec};

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
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
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
        text.add_hint(&hint);
        metadata.add_input(&text);

        let start_logits = TensorMetadata::new("start_logits");
        start_logits.set_description(
            "The indices for the start of each word/phrase to extract.",
        );
        let hint =
            supported_shapes(&[ElementType::U32], DimensionsParam::Fixed(&[0]));
        start_logits.add_hint(&hint);
        metadata.add_input(&start_logits);

        let end_logits = TensorMetadata::new("end_logits");
        end_logits.set_description(
            "The indices for the end of each word/phrase to extract.",
        );
        let hint =
            supported_shapes(&[ElementType::U32], DimensionsParam::Fixed(&[0]));
        end_logits.add_hint(&hint);
        metadata.add_input(&end_logits);

        let phrases = TensorMetadata::new("phrases");
        phrases.set_description("The phrases that were extracted.");
        let hint = supported_shapes(
            &[ElementType::Utf8],
            DimensionsParam::Fixed(&[0]),
        );
        phrases.add_hint(&hint);
        metadata.add_output(&phrases);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "text",
            ElementType::U8,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "start_logits",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "end_logits",
            ElementType::U32,
            DimensionsParam::Fixed(&[0]),
        );
        ctx.add_output_tensor(
            "phrases",
            ElementType::Utf8,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let text = ctx.get_input_tensor("text").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "text".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let start_logits =
            ctx.get_input_tensor("start_logits").ok_or_else(|| {
                KernelError::InvalidInput(InvalidInput {
                    name: "start_logits".to_string(),
                    reason: BadInputReason::NotFound,
                })
            })?;

        let end_logits =
            ctx.get_input_tensor("end_logits").ok_or_else(|| {
                KernelError::InvalidInput(InvalidInput {
                    name: "end_logits".to_string(),
                    reason: BadInputReason::NotFound,
                })
            })?;
        match text.element_type {
                ElementType::U8 =>{
                     text.buffer.view::<u8>(&text.dimensions)
                    .map_err(|e| KernelError::InvalidInput(InvalidInput{ name: "text".to_string(), reason: BadInputReason::InvalidValue(e.to_string()) }))?;
                    }
                other => {
                    return Err(KernelError::Other(format!(
                    "The Object Filter proc-block doesn't support {:?} element type",
                    other,
                    )))
                },
            };
        match start_logits.element_type {
                ElementType::U32 =>{
                    start_logits.buffer.view::<f32>(&start_logits.dimensions)
                    .map_err(|e| KernelError::InvalidInput(InvalidInput{ name: "start_logits".to_string(), reason: BadInputReason::InvalidValue(e.to_string()) }))?;
                    }
                other => {
                    return Err(KernelError::Other(format!(
                    "The Object Filter proc-block doesn't support {:?} element type",
                    other,
                    )))
                },
            };
        match end_logits.element_type {
                ElementType::U32 =>{
                    end_logits.buffer.view::<f32>(&start_logits.dimensions)
                    .map_err(|e| KernelError::InvalidInput(InvalidInput{ name: "end_logits".to_string(), reason: BadInputReason::InvalidValue(e.to_string()) }))?;
                    }
                other => {
                    return Err(KernelError::Other(format!(
                    "The Object Filter proc-block doesn't support {:?} element type",
                    other,
                    )))
                },
            };
        let output = transform((
            text.buffer.elements(),
            start_logits.buffer.elements(),
            end_logits.buffer.elements(),
        ));

        let output: Vec<String> =
            output.iter().map(|s| s.to_string()).collect();

        ctx.set_output_tensor(
            "phrases",
            TensorParam {
                element_type: ElementType::Utf8,
                dimensions: &[output.len() as u32],
                buffer: &string_tensor_from_ndarray(&ndarray::arr1(&output)),
            },
        );

        Ok(())
    }
}

fn transform<'a>(inputs: (&[u8], &[u32], &[u32])) -> Vec<String> {
    let (text, start_logits, end_logits) = inputs;

    let underlying_bytes: &[u8] = text.elements();
    let input_text = core::str::from_utf8(underlying_bytes)
        .expect("Input tensor should be valid UTF8");

    let input_text: Vec<&str> = input_text.lines().collect();

    let start_index: u32 = start_logits[0];
    let end_index: u32 = end_logits[0];
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

    let output_text = vec![(buffer)];

    println!("output {:?}", &output_text);

    output_text
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]
    fn test_token_extractor() {
        let bytes: Vec<u8> = "[UNK]\n[UNK]\nuna\n##ffa\n##ble\nworld\n!"
            .as_bytes()
            .to_vec();
        // let bytes =(bytes);
        let start_index = [2_u32];
        let end_index = [4_u32];
        let output = transform((&bytes, &start_index, &end_index));

        let should_be = vec!["unaffable".to_string()];

        assert_eq!(output, should_be);
    }
}
