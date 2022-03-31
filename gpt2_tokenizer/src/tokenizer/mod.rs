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

//!# Tokenizers
//!
//! This module contains the tokenizers to split an input text in a sequence of tokens.
//! These rely on the vocabularies for defining the subtokens a given word should be decomposed to.
//! The Byte-Pair Encoding tokenizers is implemented in this crate:
//! - Byte-Pair Encoding tokenizers:
//!     - GPT2

//!
//! All tokenizers are `Send`, `Sync` and support multi-threaded tokenization and encoding.

pub(crate) mod base_tokenizer;
mod constants;
mod gpt2_tokenizer;
pub(crate) mod tokenization_utils;

pub use base_tokenizer::{
    BaseTokenizer, MultiThreadedTokenizer, Tokenizer, TruncationStrategy,
};
pub use gpt2_tokenizer::Gpt2Tokenizer;
pub use tokenization_utils::truncate_sequences;
