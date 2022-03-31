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

//!# Vocabularies
//!
//! This module contains the vocabularies leveraged by the tokenizer. These contain methods for
//! deserialization of vocabulary files and access by the tokenizers, including:
//! - dictionaries (mapping from token to token ids)
//! - merge files (used by Byte-Pair Encoding tokenizers)
//! - sentence-piece models (trie structure and methods to find common prefix subtokens)
//!
//! The following vocabularies have been implemented:
//! - BERT
//! - ALBERT
//! - GPT2
//! - GPT
//! - Marian
//! - RoBERTa
//! - T5
//! - XLMRoBERTa
//! - XLNet
//! - SentencePiece
//!
//! All vocabularies implement the `Vocab` trait exposing a standard interface for integration with
//! the tokenizers.

pub(crate) mod base_vocab;

pub(crate) mod bpe_vocab;

mod gpt2_vocab;


pub use base_vocab::{BaseVocab, Vocab};
pub use bpe_vocab::{BpePairRef, BpePairVocab};
pub use gpt2_vocab::Gpt2Vocab;

