// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
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

use crate::alloc::borrow::ToOwned;
use crate::tokenizer::tokenization_utils::{clean_text, lowercase};
use crate::tokenizer::tokenization_utils::{
    split_on_punct, split_on_special_tokens, strip_accents, tokenize_cjk_chars,
    truncate_sequences, whitespace_tokenize,
};
use crate::vocab::Vocab;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
// use rayon::prelude::*;

/// # Truncation strategy variants
/// Indicates if and how sequence pairs exceeding a given length should be truncated
pub enum TruncationStrategy {
    /// Truncate the longest sequence first
    LongestFirst,
    /// Truncate only the first sequence
    OnlyFirst,
    /// Truncate only the second sequence
    OnlySecond,
    /// Do not truncate the sequences
    DoNotTruncate,
}

/// Crate-wide primitive used to store offset positions
pub type OffsetSize = u32;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Eq)]
///Offset information (in unicode points) to relate a token back to its original input string
pub struct Offset {
    pub begin: OffsetSize,
    pub end: OffsetSize,
}

impl Offset {
    /// Create a new offset from a begin and end positions
    pub fn new(begin: OffsetSize, end: OffsetSize) -> Offset {
        Offset { begin, end }
    }

    /// Wrap the offset into an option
    pub fn into_option(self) -> Option<Offset> {
        if self.end > self.begin {
            Some(self)
        } else {
            None
        }
    }
}

/// # Type indication for tokens (e.g. special token, white space, unknown...)
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Eq)]
pub enum Mask {
    /// The token has no particular mask. This is the default situation. It may indicate that further processing can be done on a token.
    None,
    /// The token represents a whitespace (in any shape or form)
    Whitespace,
    /// The token represents punctuation (in any shape or form)
    Punctuation,
    /// The token represents a single Chinese/Japanese/Korean character (including kana and hangul)
    CJK,
    /// The token is a special marker (such as a separator marker, a class marker, etc)
    Special,
    /// The token is the begin in a series of subtokens, the offset refers specifically to the sub-token. Subsequent tokens in this sequence will carry the 'Continuation' mask
    Begin,
    /// The token is the continuation of the previous token, the offset refers specifically to the sub-token. All but the first sub-token in a sequence carry this mask (the first carries 'Begin'). (this is the reverse of Mask::Unfinished)
    Continuation,
    /// The token is the start of a token but not finished yet. All but the last sub-token in the a token sequence carry this mask. This is the reverse of Mask::Continuation.
    Unfinished,
    /// The token is out of vocabulary, it is unknown by the tokenizer and it will decode to unknown. Tokens that can be decoded properly (but may still be out of vocabulary) should not set this.
    Unknown,
}

impl Default for Mask {
    fn default() -> Mask {
        Mask::None
    }
}

/// Token abstraction trait to access token fields, irrespective of their form (reference of owned)
pub trait TokenTrait {
    /// Returns the offset of the token with respect to the original string
    fn offset(&self) -> Option<Offset>;
    /// Returns the token mask
    fn mask(&self) -> Mask;
    /// Returns a string representation for the token
    fn as_str(&self) -> &str;
}

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
/// Reference token that references the original text, with a string slice representation
pub struct TokenRef<'a> {
    /// String representation
    pub text: &'a str,
    /// Start and end positions of the token with respect to the original text
    pub offset: Offset,
    /// Sequence of positions with respect to the original text contained in the token.
    /// For example, if the token offset is `start: 4, end: 10`, corresponding reference_offsets are `[4, 5, 6, 7, 8, 9]`
    pub reference_offsets: &'a [OffsetSize],
    /// Mask indicating the type of the token
    pub mask: Mask,
}

impl<'a> TokenRef<'a> {
    /// Creates a new token reference from a text and list of offsets.
    ///
    /// # Parameters
    /// - text (`&str`): text reference
    /// - offsets (`&[OffsetSize]`): reference positions with respect to the original text
    ///
    /// # Example
    // ```
    // use rust_tokenizers::TokenRef;
    // let _original_text = "Hello, world";
    // let text = "world";
    // let offsets = &[7, 8, 9, 10, 11];
    //
    // let token_ref = TokenRef::new(text, offsets);
    // ```
    pub fn new(text: &'a str, offsets: &'a [OffsetSize]) -> TokenRef<'a> {
        TokenRef {
            text,
            offset: Offset {
                begin: 0,
                end: offsets.len() as OffsetSize,
            },
            reference_offsets: offsets,
            mask: Mask::None,
        }
    }

    /// Converts a token reference to an owned form.
    /// # Example
    // ```
    // use rust_tokenizers::TokenRef;
    // let _original_text = "Hello, world";
    // let text = "world";
    // let offsets = &[7, 8, 9, 10, 11];
    // let token_ref = TokenRef::new(text, offsets);
    //
    // let owned_token = token_ref.to_owned();
    // ```
    pub fn to_owned(self) -> Token {
        //not a real implementation of ToOwned because that can't work in the current setup
        Token::from(self)
    }
}

impl<'a> TokenTrait for TokenRef<'a> {
    fn offset(&self) -> Option<Offset> {
        self.offset.into_option()
    }

    fn mask(&self) -> Mask {
        self.mask
    }

    fn as_str(&self) -> &str {
        self.text
    }
}

impl TokenTrait for Token {
    fn offset(&self) -> Option<Offset> {
        self.offset.into_option()
    }

    fn mask(&self) -> Mask {
        self.mask
    }

    fn as_str(&self) -> &str {
        self.text.as_str()
    }
}

impl<'a> From<&'a Token> for TokenRef<'a> {
    fn from(other: &'a Token) -> Self {
        TokenRef {
            text: other.text.as_str(),
            offset: other.offset,
            reference_offsets: &other.reference_offsets,
            mask: other.mask,
        }
    }
}

impl From<&str> for Token {
    fn from(text: &str) -> Self {
        Token::new(text.to_owned())
    }
}

impl<'a> From<TokenRef<'a>> for Token {
    fn from(other: TokenRef<'a>) -> Self {
        Token {
            text: other.text.to_owned(),
            offset: other.offset,
            reference_offsets: other.reference_offsets.to_vec(),
            mask: other.mask,
        }
    }
}

/// # ConsolidatedTokenIterator
///
/// This iterator loops over collections of tokens (implementing `TokenTrait`)
/// and groups all subtokens that belong together (forming a word or something similar).
pub struct ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    pub tokens: &'a [T],
    pub begin: usize,
    pub cursor: usize,
}

impl<'a, T> ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    /// Creates a new `ConsolidatedTokenIterator` from a sequence of `Tokens` or `TokenRefs`
    pub fn new(tokens: &'a [T]) -> Self {
        ConsolidatedTokenIterator {
            tokens,
            begin: 0,
            cursor: 0,
        }
    }
}

impl<'a, T> Iterator for ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(sub_token) = self.tokens.get(self.cursor) {
                if sub_token.mask() != Mask::Continuation {
                    //return the previous buffer of subtokens (no copies!)
                    if self.cursor > self.begin {
                        let sub_tokens = &self.tokens[self.begin..self.cursor];
                        self.begin = self.cursor;
                        self.cursor += 1;
                        return Some(sub_tokens);
                    }
                }
                self.cursor += 1;
            } else {
                //we are at past the last item, return remaining buffer
                if self.begin < self.cursor {
                    let sub_tokens = &self.tokens[self.begin..self.cursor];
                    self.cursor += 1;
                    self.begin = self.cursor;
                    return Some(sub_tokens);
                } else {
                    //nothing in buffer, we're done
                    return None;
                }
            }
        }
    }
}

/// # ConsolidatableTokens
///
/// This trait can be implemented for collections of tokens (i.e. things that implement `TokenTrait`)
/// and instantiates an iterator to quickly iterate over the tokens in consolidated form, e.g.
/// grouping subtokens into words.
///
// ```no_run
// use rust_tokenizers::{ConsolidatableTokens, Token};
// let tokens: Vec<Token> = vec![]; //add some tokens
// for (wordcount, word_tokens) in tokens.iter_consolidate_tokens().enumerate() {
//     eprintln!("word #{} - {:?}", wordcount + 1, word_tokens);
// }
// ```
pub trait ConsolidatableTokens<T>
where
    T: TokenTrait,
{
    /// Creates an iterator from a sequence of `ConsolidatableTokens`.
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<T>;
}

impl ConsolidatableTokens<Token> for Vec<Token> {
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<Token> {
        ConsolidatedTokenIterator::new(self)
    }
}

impl<'a> ConsolidatableTokens<TokenRef<'a>> for Vec<TokenRef<'a>> {
    fn iter_consolidate_tokens(
        &self,
    ) -> ConsolidatedTokenIterator<TokenRef<'a>> {
        ConsolidatedTokenIterator::new(self)
    }
}

#[derive(Debug, PartialEq, Clone)]
/// Owned token that references the original text but stores its own string representation.
pub struct Token {
    /// String representation
    pub text: String,
    /// Start and end positions of the token with respect to the original text
    pub offset: Offset,
    /// Sequence of positions with respect to the original text contained in the token.
    /// For example, if the token offset is `start: 4, end: 10`, corresponding reference_offsets are `[4, 5, 6, 7, 8, 9]`
    pub reference_offsets: Vec<OffsetSize>,
    /// Mask indicating the type of the token
    pub mask: Mask,
}

impl Token {
    /// Creates a new owned token from a `String`.
    ///
    /// # Parameters
    /// - text (`String`): text reference
    ///
    /// # Example
    // ```
    // use rust_tokenizers::Token;
    // let text = "world".to_string();
    // let token = Token::new(text);
    // ```
    pub fn new(text: String) -> Token {
        let text_size: OffsetSize = text.chars().count() as OffsetSize;
        Token {
            text,
            offset: Offset {
                begin: 0,
                end: text_size,
            },
            reference_offsets: (0..text_size).collect(),
            mask: Mask::None,
        }
    }

    // Converts an owned token to a reference form
    //
    // # Example
    // ```
    // # use rust_tokenizers::Token;
    // let text = "world".to_string();
    // let token = Token::new(text);
    //
    // let token_ref = token.as_ref();
    // ```
    pub fn as_ref(&self) -> TokenRef {
        //not a real implementation of AsRef because we do something slightly different
        TokenRef::from(self)
    }
}

/// # Tokenized Input, ready for processing in language models
/// This represents the final output of the encoding process (tokenized sentence with encoded values)
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct TokenizedInput {
    /// Vector of token IDs
    pub token_ids: Vec<i64>,

    /// Vector segments ids (for example for BERT segments are separated with a [SEP] marker, each incrementing the segment ID).
    /// This vector has the same length as token_ids.
    pub segment_ids: Vec<i8>,

    /// Flags tokens as special tokens (1) or not (0). This vector has the same length as token_ids.
    pub special_tokens_mask: Vec<i8>,

    /// Vector containing overflowing tokens, populated following a truncation step
    pub overflowing_tokens: Vec<i64>,

    /// Number of overflowing tokens following a truncation step. this equals the length `overflowing_tokens`
    pub num_truncated_tokens: usize,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub token_offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub mask: Vec<Mask>,
}

/// # Encoded input with special tokens
/// Intermediate tokenization steps before truncation to a maximum length, after encoding and addition of special tokens
#[derive(Debug, Clone)]
pub struct TokenIdsWithSpecialTokens {
    /// Vector of token IDs
    pub token_ids: Vec<i64>,

    /// Vector segments ids (for example for BERT segments are separated with a [SEP] marker, each incrementing the segment ID).
    /// This vector has the same length as token_ids.
    pub segment_ids: Vec<i8>,

    /// Flags tokens as special tokens (1) or not (0). This vector has the same length as token_ids.
    pub special_tokens_mask: Vec<i8>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub token_offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub mask: Vec<Mask>,
}

/// # Tokenized sequence
/// Intermediate tokenization steps before encoding, addition of special tokens and truncation
#[derive(Debug, Clone)]
pub struct TokensWithOffsets {
    /// Vector of token strings
    pub tokens: Vec<String>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub masks: Vec<Mask>,
}

/// # Encoded sequence
/// Intermediate tokenization steps before addition of special tokens, after encoding
#[derive(Debug, Clone, PartialEq)]
pub struct TokenIdsWithOffsets {
    /// Vector of token IDs
    pub ids: Vec<i64>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub masks: Vec<Mask>,
}

/// # Base trait for tokenizers
pub trait Tokenizer<T: Vocab> {
    /// returns a reference to the tokenizer vocabulary
    fn vocab(&self) -> &T;

    /// Tokenize a string, returns a vector of tokens as strings.
    /// Use `tokenize_with_offsets` or `tokenize_to_tokens` to return offset information.
    ///
    /// # Parameters
    /// - text : text (string-like) to tokenize
    ///
    /// # Returns
    /// `Vec<String>` containing the tokens string representation
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text = "Hello, world!";
    // let tokens = tokenizer.tokenize(text);
    // ```
    fn tokenize<S: AsRef<str>>(&self, text: S) -> Vec<String> {
        self.tokenize_with_offsets(text).tokens
    }

    /// Tokenize a string, returning tokens with offset information
    ///
    /// # Parameters
    /// - text : text (string-like) to tokenize
    ///
    /// # Returns
    /// `TokensWithOffsets` with the tokens and their offset information
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text = "Hello, world!";
    // let tokens = tokenizer.tokenize_with_offsets(text);
    // ```
    fn tokenize_with_offsets<S: AsRef<str>>(
        &self,
        text: S,
    ) -> TokensWithOffsets {
        if text.as_ref().trim().is_empty() {
            return TokensWithOffsets {
                tokens: vec![],
                offsets: vec![],
                reference_offsets: vec![],
                masks: vec![],
            };
        }
        let initial_offsets = (0..text.as_ref().chars().count() as OffsetSize)
            .collect::<Vec<OffsetSize>>();
        let initial_token: TokenRef<'_> =
            TokenRef::new(text.as_ref(), &initial_offsets);
        let tokens = self.tokenize_to_tokens(initial_token);
        let length = tokens.len();
        let mut texts = Vec::with_capacity(length);
        let mut offsets = Vec::with_capacity(length);
        let mut original_positions = Vec::with_capacity(length);
        let mut masks = Vec::with_capacity(length);

        for token in tokens {
            texts.push(token.text);
            offsets.push(if !token.reference_offsets.is_empty() {
                Some(Offset {
                    begin: *token.reference_offsets.first().unwrap(),
                    end: *token.reference_offsets.last().unwrap() + 1,
                })
            } else {
                None
            });
            original_positions.push(token.reference_offsets);
            masks.push(token.mask);
        }
        TokensWithOffsets {
            tokens: texts,
            offsets,
            reference_offsets: original_positions,
            masks,
        }
    }

    /// Tokenize a TokenRef, returning a sequence of tokens
    ///
    /// # Parameters
    /// - text (`TokenRef`): TokenRef to tokenize (this is especially useful for nested tokenization,
    /// where a tokenizer is called on the ouput of a pre-tokenizer, such as BERT).
    ///
    /// # Returns
    /// `Vec<Token>` tokenization of the original `TokenRef`
    ///
    /// # Example
    ///
    /// ```no_run
    // use itertools::Itertools;
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // use rust_tokenizers::{OffsetSize, TokenRef};
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text = "Hello, world!";
    // let offsets = (0..text.len() as OffsetSize).collect_vec();
    // let text = TokenRef::new(text, &offsets);
    // let tokens = tokenizer.tokenize_to_tokens(text);
    // ```
    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token>;

    /// Tokenize a list of strings, returning tokens with offset information
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<Vec<String>>` with the token strings representation
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let texts = ["Hello, world!", "Second sentence"];
    // let tokens = tokenizer.tokenize_list(&texts);
    // ```
    fn tokenize_list<S, ST>(&self, text_list: S) -> Vec<Vec<String>>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize(&text))
            .collect()
    }

    /// Tokenize a list of strings, where each corresponds to for example a sentence, returns a
    /// vector of TokensWithOffsets containing the tokens and their offset information. This calls
    /// `tokenize_with_offsets` on the list provided.
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<TokensWithOffsets>` with the token strings representation and offsets
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text = ["Hello, world!", "Second sentence"];
    // let tokens = tokenizer.tokenize_list_with_offsets(&text);
    // ```
    fn tokenize_list_with_offsets<S, ST>(
        &self,
        text_list: S,
    ) -> Vec<TokensWithOffsets>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize_with_offsets(text))
            .collect()
    }

    /// Convert a slice of string-like to a vector ot token indices
    ///
    /// # Parameters
    /// - tokens: list of token string-like to convert to ids
    ///
    /// # Returns
    /// `Vec<i64>` with the token indices
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let tokens = ["Hello", ",", "world", "!"];
    // let token_ids = tokenizer.convert_tokens_to_ids(&tokens);
    // ```
    fn convert_tokens_to_ids<S, ST>(&self, tokens: S) -> Vec<i64>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str>,
    {
        tokens
            .as_ref()
            .iter()
            .map(|v| self.vocab().token_to_id(v.as_ref()))
            .collect()
    }

    /// Encode a string-like (tokenization followed by encoding)
    ///
    /// # Parameters
    /// - text_1: input text (string-like) to encode
    /// - text_2: optional additional input text (string-like) to encode. When provided, both texts are
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `TokenizedInput` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask)
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text_1 = "Hello, world!";
    // let text_2 = "How is it going?";
    // let encoded_input = tokenizer.encode(
    //     text_1,
    //     Some(text_2),
    //     5,
    //     &TruncationStrategy::LongestFirst,
    //     2,
    // );
    // ```
    fn encode<S: AsRef<str>>(
        &self,
        text_1: S,
        text_2: Option<S>,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> TokenizedInput {
        let tokens = self.tokenize_with_offsets(text_1);
        let token_ids_1 = self.convert_tokens_to_ids(tokens.tokens);
        let len_1 = token_ids_1.len();
        let token_ids_with_offsets_1 = TokenIdsWithOffsets {
            ids: token_ids_1,
            offsets: tokens.offsets,
            reference_offsets: tokens.reference_offsets,
            masks: tokens.masks,
        };
        let (token_ids_with_offsets_2, len_2) = {
            if let Some(text) = text_2 {
                let tokens_2 = self.tokenize_with_offsets(text);
                let token_ids_2: Vec<i64> =
                    self.convert_tokens_to_ids(tokens_2.tokens);
                let len_2 = token_ids_2.len();
                (
                    Some(TokenIdsWithOffsets {
                        ids: token_ids_2,
                        offsets: tokens_2.offsets,
                        reference_offsets: tokens_2.reference_offsets,
                        masks: tokens_2.masks,
                    }),
                    len_2,
                )
            } else {
                (None, 0)
            }
        };
        let additional_tokens = self.build_input_with_special_tokens(
            TokenIdsWithOffsets {
                ids: vec![],
                offsets: vec![],
                reference_offsets: vec![],
                masks: vec![],
            },
            if token_ids_with_offsets_2.is_some() {
                Some(TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                })
            } else {
                None
            },
        );
        let total_len = len_1 + len_2 + additional_tokens.token_ids.len();
        let num_truncated_tokens = if total_len > max_len {
            total_len - max_len
        } else {
            0
        };
        let (
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            overflowing_tokens,
            _overflowing_offsets,
        ) = truncate_sequences(
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            num_truncated_tokens,
            truncation_strategy,
            stride,
        )
        .unwrap();

        let merged_tokenized_input = self.build_input_with_special_tokens(
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
        );

        TokenizedInput {
            token_ids: merged_tokenized_input.token_ids,
            segment_ids: merged_tokenized_input.segment_ids,
            special_tokens_mask: merged_tokenized_input.special_tokens_mask,
            overflowing_tokens,
            num_truncated_tokens,
            token_offsets: merged_tokenized_input.token_offsets,
            reference_offsets: merged_tokenized_input.reference_offsets,
            mask: merged_tokenized_input.mask,
        }
    }

    /// Encode a sequence of string-like texts (tokenization followed by encoding). Not that in contrast
    /// with `encode` optional second text, each text provided is encoded independently.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text_1 = "Hello, world!";
    // let text_2 = "How is it going?";
    // let text_3 = "Very well thank you.";
    // let encoded_input = tokenizer.encode_list(
    //     [text_1, text_2, text_3],
    //     5,
    //     &TruncationStrategy::LongestFirst,
    //     2,
    // );
    // ```
    fn encode_list<S, ST>(
        &self,
        text_list: S,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| {
                self.encode(text, None, max_len, truncation_strategy, stride)
            })
            .collect()
    }

    /// Encode a sequence of string-like text pairs (tokenization followed by encoding). This combines
    /// with `encode` with the list processing of `encode_list`.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text_1 = "Hello, world!";
    // let text_2 = "This is a second sentence";
    // let text_3 = "Very well thank you.";
    // let text_4 = "This is another second sentence.";
    // let encoded_input = tokenizer.encode_pair_list(
    //     [(text_1, text_2), (text_3, text_4)],
    //     5,
    //     &TruncationStrategy::LongestFirst,
    //     2,
    // );
    // ```
    fn encode_pair_list<S, ST>(
        &self,
        text_list: S,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<[(ST, ST)]>,
        ST: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| {
                self.encode(
                    text.0.as_ref(),
                    Some(text.1.as_ref()),
                    max_len,
                    truncation_strategy,
                    stride,
                )
            })
            .collect()
    }

    /// Cleans-up tokenization artifacts (for example whitespace before punctuation)
    ///
    /// # Arguments
    /// - input_string (`String`): input string to clean up
    ///
    /// # Returns
    /// - `String`: clean-up string
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let skip_special_tokens = true;
    // let clean_up_tokenization_spaces = true;
    // let input_string = "Hello . Do n't pay attention to the punctuation .".to_string();
    // let cleaned_string = tokenizer.clean_up_tokenization(input_string);
    // ```
    fn clean_up_tokenization(&self, input_string: String) -> String {
        input_string
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
    }

    /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    /// by concatenating and adding special tokens.
    ///
    /// For example, a RoBERTa sequence has the following format:
    /// - single sequence: <s> X </s>
    /// - pair of sequences: <s> A </s></s> B </s>
    ///
    /// # Parameters
    /// - tokens_ids_with_offsets_1 (`TokenIdsWithOffsets`): first sequence
    /// - tokens_ids_with_offsets_2 (`TokenIdsWithOffsets`): (optional) second sequence
    ///
    /// # Returns
    /// - `TokenIdsWithSpecialTokens` containing a concatenation of both sequences with added special tokens
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // use rust_tokenizers::TokenIdsWithOffsets;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let skip_special_tokens = true;
    // let clean_up_tokenization_spaces = true;
    // let first_sequence = "Hello, world";
    // let second_sequence = "This is the second sentence";
    //
    // let first_tokens = tokenizer.tokenize_with_offsets(first_sequence);
    // let first_ids = tokenizer.convert_tokens_to_ids(first_tokens.tokens);
    // let first_input = TokenIdsWithOffsets {
    //     ids: first_ids,
    //     offsets: first_tokens.offsets,
    //     reference_offsets: first_tokens.reference_offsets,
    //     masks: first_tokens.masks,
    // };
    //
    // let second_tokens = tokenizer.tokenize_with_offsets(second_sequence);
    // let second_ids = tokenizer.convert_tokens_to_ids(second_tokens.tokens);
    // let second_input = TokenIdsWithOffsets {
    //     ids: second_ids,
    //     offsets: second_tokens.offsets,
    //     reference_offsets: second_tokens.reference_offsets,
    //     masks: second_tokens.masks,
    // };
    //
    // let combined_with_special_tokens =
    //     tokenizer.build_input_with_special_tokens(first_input, Some(second_input));
    // ```
    fn build_input_with_special_tokens(
        &self,
        mut tokens_ids_with_offsets_1: TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenIdsWithSpecialTokens {
        let mut token_segment_ids: Vec<i8> =
            vec![0; tokens_ids_with_offsets_1.ids.len()];
        let mut special_tokens_mask: Vec<i8> =
            vec![0; tokens_ids_with_offsets_1.ids.len()];
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2
        {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            token_segment_ids.extend(vec![1; length]);
            special_tokens_mask.extend(vec![0; length]);
            tokens_ids_with_offsets_1
                .ids
                .extend(tokens_ids_with_offsets_2_value.ids);
            tokens_ids_with_offsets_1
                .offsets
                .extend(tokens_ids_with_offsets_2_value.offsets);
            tokens_ids_with_offsets_1
                .reference_offsets
                .extend(tokens_ids_with_offsets_2_value.reference_offsets);
            tokens_ids_with_offsets_1
                .masks
                .extend(tokens_ids_with_offsets_2_value.masks);
        };

        TokenIdsWithSpecialTokens {
            token_ids: tokens_ids_with_offsets_1.ids,
            segment_ids: token_segment_ids,
            special_tokens_mask,
            token_offsets: tokens_ids_with_offsets_1.offsets,
            reference_offsets: tokens_ids_with_offsets_1.reference_offsets,
            mask: tokens_ids_with_offsets_1.masks,
        }
    }
}

/// # Extension for multithreaded tokenizers
pub trait MultiThreadedTokenizer<T: Vocab>
where
    Self: Sync + Send + Tokenizer<T>,
{
    /// returns a reference to the tokenizer vocabulary
    fn vocab(&self) -> &T {
        Tokenizer::<T>::vocab(self)
    }

    /// Tokenize a list of strings (with multithreading), where each corresponds to for example a sentence, returns a
    /// vector of TokensWithOffsets containing the tokens and their offset information. This calls
    /// `tokenize_with_offsets` on the list provided.
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<TokensWithOffsets>` with the token strings representation and offsets
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text = ["Hello, world!", "Second sentence"];
    // let tokens = tokenizer.tokenize_list_with_offsets(&text);
    // ```
    fn tokenize_list_with_offsets<S, ST>(
        &self,
        text_list: S,
    ) -> Vec<TokensWithOffsets>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str> + Sync,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize_with_offsets(text))
            .collect()
    }

    /// Multithreaded tokenization of a list of strings, returning tokens with offset information
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<Vec<String>>` with the token strings representation
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, MultiThreadedTokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let texts = ["Hello, world!", "Second sentence"];
    // let tokens = tokenizer.tokenize_list(&texts);
    // ```
    fn tokenize_list<S, ST>(&self, text_list: S) -> Vec<Vec<String>>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str> + Sync,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize(text))
            .collect()
    }

    /// Multithreaded encoding of a sequence of string-like texts (tokenization followed by encoding). Not that in contrast
    /// with `encode` optional second text, each text provided is encoded independently.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, MultiThreadedTokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text_1 = "Hello, world!";
    // let text_2 = "How is it going?";
    // let text_3 = "Very well thank you.";
    // let encoded_input = tokenizer.encode_list(
    //     [text_1, text_2, text_3],
    //     5,
    //     &TruncationStrategy::LongestFirst,
    //     2,
    // );
    // ```
    fn encode_list<S, ST>(
        &self,
        text_list: S,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<[ST]>,
        ST: AsRef<str> + Sync,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| {
                self.encode(text, None, max_len, truncation_strategy, stride)
            })
            .collect()
    }

    /// Multithreaded ncoding of a sequence of string-like text pairs (tokenization followed by encoding). This combines
    /// with `encode` with the list processing of `encode_list`.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, MultiThreadedTokenizer, TruncationStrategy};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    //
    // let text_1 = "Hello, world!";
    // let text_2 = "This is a second sentence";
    // let text_3 = "Very well thank you.";
    // let text_4 = "This is another second sentence.";
    // let encoded_input = tokenizer.encode_pair_list(
    //     [(text_1, text_2), (text_3, text_4)],
    //     5,
    //     &TruncationStrategy::LongestFirst,
    //     2,
    // );
    // ```
    fn encode_pair_list<S, ST>(
        &self,
        text_list: S,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<[(ST, ST)]>,
        ST: AsRef<str> + Sync,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| {
                self.encode(
                    text.0.as_ref(),
                    Some(text.1.as_ref()),
                    max_len,
                    truncation_strategy,
                    stride,
                )
            })
            .collect()
    }
}

/// # Base tokenizer
/// Base tokenizer performing:
/// - whitespace tokenization
/// - splitting on special characters
/// - splitting on punctuation
/// - splitting on CJK characters
/// - (optional) lower casing
/// - (optional) accent stripping
///
/// This tokenizer is used as a pre-tokenizer step in the BERT and GPT tokenizers.
pub struct BaseTokenizer<T: Vocab> {
    vocab: T,
    lower_case: bool,
    strip_accents: bool,
}

impl<T: Vocab + Sync + Send> BaseTokenizer<T> {
    /// Create a new instance of a `BaseTokenizer`
    /// Expects a vocabulary flat-file as an input.
    ///
    /// # Parameters
    /// - path (`&str`): path to the vocabulary file (only used for special character splitting)
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::BaseVocab;
    // let strip_accents = false;
    // let lower_case = false;
    // let tokenizer: BaseTokenizer<BaseVocab> =
    //     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    // ```

    // pub fn from_file(
    //     path: &str,
    //     lower_case: bool,
    //     strip_accents: bool,
    // ) -> Result<BaseTokenizer<T>,TokenizerError> {
    //     let vocab = T::from_file(path)?;
    //     Ok(BaseTokenizer {
    //         vocab,
    //         lower_case,
    //         strip_accents,
    //     })
    // }

    /// Create a new instance of a `BaseTokenizer` from an existing vocabulary
    ///
    /// # Parameters
    /// - vocab (`Vocab`): Thread-safe reference to a vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    /// - strip_accents (`bool`): flag indicating if accents should be stripped from the text
    ///
    /// # Example
    ///
    // ```no_run
    // use rust_tokenizers::tokenizer::{BaseTokenizer, Tokenizer};
    // use rust_tokenizers::vocab::{BaseVocab, Vocab};
    // let strip_accents = false;
    // let lower_case = false;
    // let base_vocab = BaseVocab::from_file("path/to/vocab/file").unwrap();
    //
    // let tokenizer = BaseTokenizer::from_existing_vocab(base_vocab, lower_case, strip_accents);
    // ```
    pub fn from_existing_vocab(
        vocab: T,
        lower_case: bool,
        strip_accents: bool,
    ) -> BaseTokenizer<T> {
        BaseTokenizer {
            vocab,
            lower_case,
            strip_accents,
        }
    }
}

impl<T: Vocab + Sync + Send> Tokenizer<T> for BaseTokenizer<T> {
    fn vocab(&self) -> &T {
        &self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        //split on whitespace
        let tokens: Vec<Token> = whitespace_tokenize(initial_token)
            .into_iter()
            .map(|token| {
                //split on special tokens
                split_on_special_tokens(token, &self.vocab)
            })
            .flatten()
            .map(|token| {
                //split on punctuation (with care for maintaining special values)
                split_on_punct(token)
            })
            .flatten()
            .map(|token| {
                //tokenize CJK characters so each character is one token
                tokenize_cjk_chars(token)
            })
            .flatten()
            .map(|token| {
                // v-- this is where the token gets owned, all steps above handle TokenRefs (dealing with &str)
                let mut token = Token {
                    text: token.text.to_string(),
                    offset: token.offset,
                    reference_offsets: token.reference_offsets.to_vec(),
                    mask: token.mask,
                };
                if token.mask != Mask::Special && token.mask != Mask::Unknown {
                    clean_text(&mut token, true);
                    //apply the necessary transformations to the actual tokens (unless it's a special value)
                    if self.lower_case {
                        lowercase(&mut token);
                    }
                    if self.strip_accents {
                        strip_accents(&mut token);
                    }
                }
                token
            })
            .filter(|token| !token.text.is_empty())
            .collect();

        tokens
    }
}

impl<T: Vocab + Sync + Send> MultiThreadedTokenizer<T> for BaseTokenizer<T> {}
