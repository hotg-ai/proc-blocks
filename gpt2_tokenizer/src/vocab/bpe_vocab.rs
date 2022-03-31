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

use alloc::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};

/// # Byte pair query
/// Structure holding a pair of bytes for query in the BPE vocabulary
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct BpePairRef<'a> {
    pub byte_1: &'a String,
    pub byte_2: &'a String,
}

/// # Byte pair Encoding Vocab
/// BPE vocab containing the merges (dictionary of pairs with their priority) used to merge
/// pairs together. This vocabulary element is used on BPE tokenizers such as GPT2 or RoBERTa.
/// This vocabulary is not meant to be used directly, but rather as part of a BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct BpePairVocab {
    pub values: BTreeMap<(String, String), i64>,
}

impl BpePairVocab {
    /// Create a new `BpePairVocab` from a flat file containing merges in the format `first elment second element`)
    /// The indices are implied by the lien position of each pair in the merges file. The first line needs to be a
    /// header and is skipped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let bpe_vocab = BpePairVocab::from_file(path);
    /// ```
    pub fn from_file(path: &str) -> Result<BpePairVocab, TokenizerError> {
        let f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let br = BufReader::new(f);
        let mut data = HashMap::new();
        let mut index = 0;
        for line in br.lines().skip(1) {
            let line = match line {
                Ok(value) => value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            };
            let tuple: Vec<String> = line.trim().split(' ').map(|v| v.to_owned()).collect();
            if tuple.len() > 1 {
                data.insert((tuple[0].clone(), tuple[1].clone()), index);
                index += 1;
            }
        }

        Ok(BpePairVocab { values: data })
    }

    /// Create a new `BpePairVocab` from a SentencePiece file containing a BPE model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairVocab, Vocab};
    /// let path = "path/to/spiece.model";
    ///
    /// let bpe_vocab = BpePairVocab::from_sentencepiece_file(path);
    /// ```
    pub fn from_sentencepiece_file(path: &str) -> Result<BpePairVocab, TokenizerError> {
        let mut f = File::open(path).map_err(|e| {
            TokenizerError::FileNotFound(format!("{} vocabulary file not found :{}", path, e))
        })?;
        let mut contents = Vec::new();
        let proto = match f.read_to_end(&mut contents) {
            Ok(_) => match ModelProto::parse_from_bytes(contents.as_slice()) {
                Ok(proto_value) => proto_value,
                Err(e) => {
                    return Err(TokenizerError::VocabularyParsingError(e.to_string()));
                }
            },
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };
        let mut values = BTreeMap::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            values.insert(piece.get_piece().to_owned(), idx as i64);
        }

        let mut data = BTreeMap::new();
        for l_piece in proto.get_pieces().iter().map(|v| v.get_piece()) {
            for r_piece in proto.get_pieces().iter().map(|v| v.get_piece()) {
                if let Some(id) = values.get(&[l_piece, r_piece].concat()) {
                    data.insert((l_piece.to_string(), r_piece.to_string()), *id);
                }
            }
        }

        Ok(BpePairVocab { values: data })
    }

    /// Gets the id of a "byte pair" in the merges vocab. Returns an optional index for the pair if
    /// it is found in the vocabulary.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_tokenizers::vocab::{BpePairRef, BpePairVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let bpe_vocab = BpePairVocab::from_file(path).unwrap();
    ///
    /// let query = BpePairRef {
    ///     byte_1: &"won".to_string(),
    ///     byte_2: &"derful".to_string(),
    /// };
    /// let id = bpe_vocab.byte_pair_to_id(&query);
    /// ```
    pub fn byte_pair_to_id(&self, byte_pair: &BpePairRef) -> Option<&i64> {
        unsafe {
            let byte_1 = byte_pair.byte_1;
            let byte_2 = byte_pair.byte_2;
            let k = (ptr::read(byte_1), ptr::read(byte_2));
            let k = ManuallyDrop::new(k);
            self.values.get(&k)
        }
    }
}
