// Copyright 2018 The Open AI Team Authors
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

use crate::error::TokenizerError;
use crate::vocab::base_vocab::{swap_key_values, Vocab};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// # GPT2 Vocab
/// Vocabulary for GPT2 tokenizer. Contains the following special values:
/// - BOS token
/// - EOS token
///
/// Expects a JSON-format vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct Gpt2Vocab {
    /// A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    /// A mapping of token ids to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    /// The string to use for unknown (out of vocabulary) tokens
    pub unknown_value: &'static str,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

impl Gpt2Vocab {
    /// Returns the BOS token for GPT2 (`<|endoftext|>`)
    pub fn bos_value() -> &'static str {
        "<|endoftext|>"
    }

    /// Returns the EOS token for GPT2 (`<|endoftext|>`)
    pub fn eos_value() -> &'static str {
        "<|endoftext|>"
    }
}

impl Vocab for Gpt2Vocab {
    fn unknown_value() -> &'static str {
        "<|endoftext|>"
    }

    fn get_unknown_value(&self) -> &'static str {
        "<|endoftext|>"
    }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn special_indices(&self) -> &HashMap<i64, String> {
        &self.special_indices
    }

    fn from_file(path: &str) -> Result<Gpt2Vocab, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!(
                "{} vocabulary file not found :{}",
                path, e
            ))
        })?;
        let br = BufReader::new(f);
        let values: HashMap<String, i64> = match serde_json::from_reader(br) {
            Ok(value) => value,
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(
                    e.to_string(),
                ));
            },
        };
        let mut special_values = HashMap::new();
        let unknown_value = Gpt2Vocab::unknown_value();
        Gpt2Vocab::_register_as_special_value(
            unknown_value,
            &values,
            &mut special_values,
        )?;

        let bos_value = Gpt2Vocab::bos_value();
        Gpt2Vocab::_register_as_special_value(
            bos_value,
            &values,
            &mut special_values,
        )?;

        let eos_value = Gpt2Vocab::eos_value();
        Gpt2Vocab::_register_as_special_value(
            eos_value,
            &values,
            &mut special_values,
        )?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);

        Ok(Gpt2Vocab {
            values,
            indices,
            unknown_value,
            special_values,
            special_indices,
        })
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_values,
            self.unknown_value,
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            id,
            &self.indices,
            &self.special_indices,
            self.unknown_value,
        )
    }
}
