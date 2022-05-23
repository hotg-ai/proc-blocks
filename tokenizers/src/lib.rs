use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    runtime_v1::{
        register_node, supported_shapes, DimensionsParam, ElementType,
        GraphContext, KernelContext, Metadata, TensorMetadata, TensorParam,
    },
    BufferExt, SliceExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Tokenizers", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "Tokenize a question and a paragraph using the Bert tokenizer.",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("nlp");
        metadata.add_tag("bert");
        metadata.add_tag("tokenization");

        let question = TensorMetadata::new("question");
        let hint =
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
        question.add_hint(&hint);
        metadata.add_input(&question);

        let paragraph = TensorMetadata::new("paragraph");
        let hint =
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
        paragraph.add_hint(&hint);
        metadata.add_input(&paragraph);

        let token_ids = TensorMetadata::new("token_ids");
        token_ids.set_description("The IDs for each token in the input.");
        let hint = supported_shapes(
            &[ElementType::I32],
            DimensionsParam::Fixed(&[1, 384]),
        );
        token_ids.add_hint(&hint);
        metadata.add_output(&token_ids);

        let token_mask = TensorMetadata::new("token_mask");
        token_mask.set_description("A set of masks indicating whether an input token is inside a segment or not.");
        let hint = supported_shapes(
            &[ElementType::I32],
            DimensionsParam::Fixed(&[1, 384]),
        );
        token_mask.add_hint(&hint);
        metadata.add_output(&token_mask);

        let segment_ids = TensorMetadata::new("segment_ids");
        segment_ids.set_description("The ID of the segment each token is in.");
        let hint = supported_shapes(
            &[ElementType::I32],
            DimensionsParam::Fixed(&[1, 384]),
        );
        segment_ids.add_hint(&hint);
        metadata.add_output(&segment_ids);

        let encoded_text = TensorMetadata::new("encoded_text");
        encoded_text.set_description(
            "The encoded question and paragraph that was fed to the tokenizer.",
        );
        let hint =
            supported_shapes(&[ElementType::U8], DimensionsParam::Fixed(&[0]));
        encoded_text.add_hint(&hint);
        metadata.add_output(&encoded_text);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "question",
            ElementType::U8,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "paragraph",
            ElementType::U8,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
            "token_ids",
            ElementType::I32,
            DimensionsParam::Fixed(&[1, 384]),
        );
        ctx.add_output_tensor(
            "token_mask",
            ElementType::I32,
            DimensionsParam::Fixed(&[1, 384]),
        );
        ctx.add_output_tensor(
            "segment_ids",
            ElementType::I32,
            DimensionsParam::Fixed(&[1, 384]),
        );

        ctx.add_output_tensor(
            "encoded_text",
            ElementType::U8,
            DimensionsParam::Fixed(&[1, 384]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let question = ctx.get_input_tensor("question").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "question".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let paragraph = ctx.get_input_tensor("paragraph").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "paragraph".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        match question.element_type {
            ElementType::U8 => {
                question.buffer.view::<u8>(&question.dimensions).map_err(
                    |e| {
                        KernelError::InvalidInput(InvalidInput {
                            name: "question".to_string(),
                            reason: BadInputReason::InvalidValue(e.to_string()),
                        })
                    },
                )?;
            },
            other => {
                return Err(KernelError::Other(format!(
                "The Tokenizer proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        match paragraph.element_type {
            ElementType::U8 => {
                paragraph.buffer.view::<u8>(&paragraph.dimensions).map_err(
                    |e| {
                        KernelError::InvalidInput(InvalidInput {
                            name: "paragraph".to_string(),
                            reason: BadInputReason::InvalidValue(e.to_string()),
                        })
                    },
                )?;
            },
            other => {
                return Err(KernelError::Other(format!(
                "The Tokenizer proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        let output = transform((
            question.buffer.elements(),
            paragraph.buffer.elements(),
        ));

        ctx.set_output_tensor(
            "token_ids",
            TensorParam {
                element_type: ElementType::I32,
                dimensions: &[1, output.0.len() as u32],
                buffer: &output.0.as_bytes(),
            },
        );
        ctx.set_output_tensor(
            "token_mask",
            TensorParam {
                element_type: ElementType::I32,
                dimensions: &[1, output.1.len() as u32],
                buffer: &output.1.as_bytes(),
            },
        );
        ctx.set_output_tensor(
            "segment_ids",
            TensorParam {
                element_type: ElementType::I32,
                dimensions: &[1, output.2.len() as u32],
                buffer: &output.2.as_bytes(),
            },
        );
        ctx.set_output_tensor(
            "encoded_text",
            TensorParam {
                element_type: ElementType::U8,
                dimensions: &[1, output.3.len() as u32],
                buffer: &output.3.as_bytes(),
            },
        );

        Ok(())
    }
}

#[macro_use]
extern crate alloc;

pub mod tokenizer;
pub mod vocab;

use crate::{
    tokenizer::{
        base_tokenizer::{
            Mask, Offset, OffsetSize, Token, TokenRef, TokenizedInput,
        },
        BertTokenizer, Tokenizer, TruncationStrategy,
    },
    vocab::{BertVocab, Vocab},
};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::str::FromStr;

pub struct Tokenizers {
    bert_tokenizer: BertTokenizer,
    bert_vocab: BertVocab,
}

impl Default for Tokenizers {
    fn default() -> Tokenizers {
        let vocabulary_text = include_str!("bert-base-uncased-vocab.txt");

        let vocab = BertVocab::from_str(vocabulary_text).unwrap();
        let vocab_copy = vocab.clone();
        let bert_tokenizer =
            BertTokenizer::from_existing_vocab(vocab, true, true);

        Tokenizers {
            bert_tokenizer,
            bert_vocab: vocab_copy,
        }
    }
}

fn transform(s: (&[u8], &[u8])) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<u8>) {
    let (s1, s2) = s;
    let underlying_bytes_1: &[u8] = s1.elements();
    let input_text_1: &str = core::str::from_utf8(underlying_bytes_1)
        .expect("Input tensor should be valid UTF8");
    let input_text_1 = input_text_1.trim_end_matches('\0');
    assert!(!input_text_1.is_empty(), "Sentence 1 is empty");
    let underlying_bytes_2: &[u8] = s2.elements();
    let input_text_2: &str = core::str::from_utf8(underlying_bytes_2)
        .expect("Input tensor should be valid UTF8");
    let input_text_2 = input_text_2.trim_end_matches('\0');
    assert!(!input_text_2.is_empty(), "Sentence 2 is empty");

    let tok: Tokenizers = Default::default();

    let TokenizedInput {
        mut token_ids,
        special_tokens_mask: _,
        mut segment_ids,
        ..
    } = tok.bert_tokenizer.encode(
        input_text_1,
        Some(input_text_2),
        384,
        &TruncationStrategy::LongestFirst,
        0,
    );

    let mut mask_ids: Vec<i32> = vec![1; token_ids.len()];
    token_ids.resize(384, 0);
    mask_ids.resize(384, 0);
    segment_ids.resize(384, 0);

    let input_ids: Vec<i32> =
        token_ids.iter().map(|&x| x as i32).collect::<Vec<i32>>();

    let seg_ids: Vec<i32> =
        segment_ids.iter().map(|&x| x as i32).collect::<Vec<i32>>();

    let mut words = String::new();
    let tok_ids = &token_ids[0 as usize..];

    for id in tok_ids {
        let s = tok.bert_vocab.id_to_token(*id);

        words.push_str(&s);
        words.push_str("\n");
    }
    words = words.to_string();
    let words: Vec<u8> = words.as_bytes().to_vec();

    (input_ids, mask_ids, seg_ids, words)
}

#[cfg(test)]

mod tests {
    use super::*;
    #[test]
    fn test_input_ids() {
        let word1: Vec<u8> = "What is Google?".as_bytes().to_vec();

        let word2: Vec<u8> =
            "Google LLC is an American multinational technology company."
                .as_bytes()
                .to_vec();
        let (input_ids, _mask_ids, _segment_ids, _word_bytes) =
            transform((&word1, &word2));

        let input_ids_should_be = vec![
            101, 2054, 2003, 8224, 1029, 102, 8224, 11775, 2003, 2019, 2137,
            20584, 2974, 2194, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        assert_eq!(input_ids, input_ids_should_be);
    }

    #[test]
    fn test_mask_ids() {
        let word1: Vec<u8> = "What is Google?".as_bytes().to_vec();

        let word2: Vec<u8> =
            "Google LLC is an American multinational technology company."
                .as_bytes()
                .to_vec();
        let (_input_ids, mask_ids, _segment_ids, _word_bytes) =
            transform((&word1, &word2));

        let mask_ids_should_be = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        assert_eq!(mask_ids, mask_ids_should_be);
    }

    #[test]
    fn test_segment_ids() {
        let word1: Vec<u8> = "What is Google?".as_bytes().to_vec();

        let word2: Vec<u8> =
            "Google LLC is an American multinational technology company."
                .as_bytes()
                .to_vec();

        let (_input_ids, _mask_ids, segment_ids, _word_bytes) =
            transform((&word1, &word2));

        let segment_ids_should_be = vec![
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        assert_eq!(segment_ids, segment_ids_should_be);
    }

    #[test]
    fn test_word_bytes() {
        let word1: Vec<u8> = "What is Google?".as_bytes().to_vec();

        let word2: Vec<u8> =
            "Google LLC is an American multinational technology company."
                .as_bytes()
                .to_vec();
        let (_input_ids, _mask_ids, _segment_ids, word_bytes) =
            transform((&word1, &word2));

        let word_bytes_should_be = vec![
            91, 67, 76, 83, 93, 10, 119, 104, 97, 116, 10, 105, 115, 10, 103,
            111, 111, 103, 108, 101, 10, 63, 10, 91, 83, 69, 80, 93, 10, 103,
            111, 111, 103, 108, 101, 10, 108, 108, 99, 10, 105, 115, 10, 97,
            110, 10, 97, 109, 101, 114, 105, 99, 97, 110, 10, 109, 117, 108,
            116, 105, 110, 97, 116, 105, 111, 110, 97, 108, 10, 116, 101, 99,
            104, 110, 111, 108, 111, 103, 121, 10, 99, 111, 109, 112, 97, 110,
            121, 10, 46, 10, 91, 83, 69, 80, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80,
            65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91,
            80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10,
            91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93,
            10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68,
            93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65, 68, 93, 10, 91, 80, 65,
            68, 93, 10, 91, 80, 65, 68, 93, 10,
        ];

        assert_eq!(word_bytes, word_bytes_should_be);
    }

    #[test]
    #[should_panic(expected = "Sentence 1 is empty")]
    fn empty_sentence_1() {
        let word1: Vec<u8> = "".as_bytes().to_vec();
        let word2: Vec<u8> = "Hi".as_bytes().to_vec();
        let (_input_ids, _mask_ids, _segment_ids, _word_bytes) =
            transform((&word1, &word2));
    }

    #[test]
    #[should_panic(expected = "Sentence 2 is empty")]
    fn empty_sentence_2() {
        let word1: Vec<u8> = "Hi".as_bytes().to_vec();

        let word2: Vec<u8> = "".as_bytes().to_vec();

        let (_input_ids, _mask_ids, _segment_ids, _word_bytes) =
            transform((&word1, &word2));
    }
}
