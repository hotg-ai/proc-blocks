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
//!
//! The following vocabularies have been implemented:
//! - BERT
//! Vocabulary implement the `Vocab` trait exposing a standard interface for integration with

pub(crate) mod base_vocab;
pub mod bert_vocab;

pub use base_vocab::Vocab;
pub use bert_vocab::BertVocab;
