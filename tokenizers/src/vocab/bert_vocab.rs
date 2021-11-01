// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::alloc::string::ToString;
use crate::vocab::base_vocab::{swap_key_values, Vocab};
use alloc::{collections::BTreeMap, string::String};
use anyhow::Result;
use core::str::FromStr;

#[derive(Debug, Clone)]
pub enum TokenError {
    TokenNotFound { word: String },
}
/// # BERT Vocab
/// Vocabulary for BERT tokenizer. Contains the following special values:
/// - CLS token
/// - SEP token
/// - PAD token
/// - MASK token
///
/// Expects a flat text vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct BertVocab {
    pub values: BTreeMap<String, i64>,
    pub indices: BTreeMap<i64, String>,
    pub special_value_indices: BTreeMap<String, i64>,
    pub special_indices: BTreeMap<i64, String>,
}

impl BertVocab {
    /// Returns the PAD token for BERT (`[PAD]`)
    pub const PAD: &'static str = "[PAD]";

    /// Returns the SEP token for BERT (`[SEP]`)
    pub const SEPARATOR: &'static str = "[SEP]";

    /// Returns the CLS token for BERT (`[CLS]`)
    pub const CLS: &'static str = "[CLS]";

    /// Returns the MASK token for BERT (`[MASK]`)
    pub const MASK: &'static str = "[MASK]";

    /// Returns the MASK token for BERT (`[UNK]`)
    pub const UNKNOWN: &'static str = "[UNK]";

    pub const SPECIAL_VALUES: &'static [&'static str] = &[
        BertVocab::UNKNOWN,
        BertVocab::MASK,
        BertVocab::SEPARATOR,
        BertVocab::CLS,
        BertVocab::PAD,
    ];
}

#[derive(Debug, Clone)]
pub enum ParseError {
    DuplicateWord {
        word: String,
        original_index: i64,
        duplicate_index: i64,
    },
}

impl FromStr for BertVocab {
    type Err = ParseError;

    fn from_str(dictionary: &str) -> Result<Self, ParseError> {
        let mut values = BTreeMap::new();
        let mut next_index = 0;

        for line in dictionary.lines() {
            let word = line.trim();

            if let Some(original_index) =
                values.insert(word.to_string(), next_index)
            {
                return Err(ParseError::DuplicateWord {
                    word: word.to_string(),
                    original_index,
                    duplicate_index: next_index,
                });
            }

            next_index += 1;
        }

        let mut special_value_indices = BTreeMap::new();

        let unknown_value = BertVocab::UNKNOWN;
        BertVocab::_register_as_special_value(
            unknown_value,
            &values,
            &mut special_value_indices,
        )
        .expect("Token index not found in vocabulary");

        let pad_value = BertVocab::PAD;
        BertVocab::_register_as_special_value(
            pad_value,
            &values,
            &mut special_value_indices,
        )
        .expect("Token index not found in vocabulary");

        let sep_value = BertVocab::SEPARATOR;
        BertVocab::_register_as_special_value(
            sep_value,
            &values,
            &mut special_value_indices,
        )
        .expect("Token index not found in vocabulary");

        let cls_value = BertVocab::CLS;
        BertVocab::_register_as_special_value(
            cls_value,
            &values,
            &mut special_value_indices,
        )
        .expect("Token index not found in vocabulary");

        let mask_value = BertVocab::MASK;
        BertVocab::_register_as_special_value(
            mask_value,
            &values,
            &mut special_value_indices,
        )
        .expect("Token index not found in vocabulary");

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_value_indices);

        Ok(BertVocab {
            values,
            indices,
            special_value_indices,
            special_indices,
        })
    }
}

impl Vocab for BertVocab {
    fn values(&self) -> &BTreeMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &BTreeMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &BTreeMap<String, i64> {
        &self.special_value_indices
    }

    fn special_indices(&self) -> &BTreeMap<i64, String> {
        &self.special_indices
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_value_indices,
            "[UNK]",
        )
    }

    fn id_to_token(&self, id: i64) -> &str {
        self._id_to_token(id, &self.indices, &self.special_indices, "[UNK]")
    }
}
