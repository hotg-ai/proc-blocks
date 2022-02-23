// Copyright 2018 The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{
    tokenizer::{
        base_tokenizer::{
            BaseTokenizer, Mask, MultiThreadedTokenizer, Offset, OffsetSize,
            Token, TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef,
            Tokenizer,
        },
        tokenization_utils::tokenize_wordpiece,
    },
    vocab::{BertVocab, Vocab},
};
use alloc::vec::Vec;

/// # BERT tokenizer
/// BERT tokenizer performing:
/// - BaseTokenizer tokenization (see `BaseTokenizer` for more details)
/// - WordPiece tokenization
pub struct BertTokenizer {
    vocab: BertVocab,
    base_tokenizer: BaseTokenizer<BertVocab>,
}

impl BertTokenizer {
    // Create a new instance of a `BertTokenizer` from an existing vocabulary
    //
    // # Parameters
    // - vocab (`BertVocab`): Thread-safe reference to a BERT vocabulary
    // - lower_case (`bool`): flag indicating if the text should be lower-cased
    //   as part of the tokenization
    // - strip_accents (`bool`): flag indicating if accents should be stripped
    //   from the text
    //
    // # Example
    //
    // ```no_run
    // use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::{BertVocab, Vocab};
    // let strip_accents = false;
    // let lower_case = false;
    // let vocab = BertVocab::from_file("path/to/vocab/file").unwrap();
    //
    // let tokenizer = BertTokenizer::from_existing_vocab(vocab, lower_case, strip_accents);
    // ```

    pub fn from_existing_vocab(
        vocab: BertVocab,
        lower_case: bool,
        strip_accents: bool,
    ) -> BertTokenizer {
        let base_tokenizer = BaseTokenizer::from_existing_vocab(
            vocab.clone(),
            lower_case,
            strip_accents,
        );
        BertTokenizer {
            vocab,
            base_tokenizer,
        }
    }
}

impl Tokenizer<BertVocab> for BertTokenizer {
    fn vocab(&self) -> &BertVocab { &self.vocab }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        // the base tokenizers does most of the work, we simply add a wordpiece
        // tokenizer on top
        self.base_tokenizer
            .tokenize_to_tokens(initial_token)
            .into_iter()
            .map(|token| tokenize_wordpiece(token.as_ref(), &self.vocab, 100))
            .flatten()
            .collect()
    }

    // fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
    //     tokens.join(" ").replace(" ##", "").trim().to_owned()
    // }

    fn build_input_with_special_tokens(
        &self,
        tokens_ids_with_offsets_1: TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenIdsWithSpecialTokens {
        let mut output: Vec<i64> = vec![];
        let mut token_segment_ids: Vec<i8> = vec![];
        let mut special_tokens_mask: Vec<i8> = vec![];
        let mut offsets: Vec<Option<Offset>> = vec![];
        let mut original_offsets: Vec<Vec<OffsetSize>> = vec![];
        let mut mask: Vec<Mask> = vec![];
        special_tokens_mask.push(1);
        special_tokens_mask
            .extend(vec![0; tokens_ids_with_offsets_1.ids.len()]);
        special_tokens_mask.push(1);
        token_segment_ids
            .extend(vec![0; tokens_ids_with_offsets_1.ids.len() + 2]);
        output.push(self.vocab.token_to_id(BertVocab::CLS));
        output.extend(tokens_ids_with_offsets_1.ids);
        output.push(self.vocab.token_to_id(BertVocab::SEPARATOR));
        offsets.push(None);
        offsets.extend(tokens_ids_with_offsets_1.offsets);
        offsets.push(None);
        original_offsets.push(vec![]);
        original_offsets.extend(tokens_ids_with_offsets_1.reference_offsets);
        original_offsets.push(vec![]);
        mask.push(Mask::Special);
        mask.extend(tokens_ids_with_offsets_1.masks);
        mask.push(Mask::Special);
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2
        {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            special_tokens_mask.extend(vec![0; length]);
            special_tokens_mask.push(1);
            token_segment_ids.extend(vec![1; length + 1]);
            output.extend(tokens_ids_with_offsets_2_value.ids);
            output.push(self.vocab.token_to_id(BertVocab::SEPARATOR));
            offsets.extend(tokens_ids_with_offsets_2_value.offsets);
            original_offsets
                .extend(tokens_ids_with_offsets_2_value.reference_offsets);
            offsets.push(None);
            original_offsets.push(vec![]);
            mask.extend(tokens_ids_with_offsets_2_value.masks);

            mask.push(Mask::Special);
        }
        TokenIdsWithSpecialTokens {
            token_ids: output,
            segment_ids: token_segment_ids,
            special_tokens_mask,
            token_offsets: offsets,
            reference_offsets: original_offsets,
            mask,
        }
    }
}

impl MultiThreadedTokenizer<BertVocab> for BertTokenizer {}
