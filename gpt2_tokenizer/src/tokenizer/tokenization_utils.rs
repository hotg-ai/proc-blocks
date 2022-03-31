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

use crate::vocab::bpe_vocab::{BpePairRef, BpePairVocab};
use crate::{
    tokenizer::{
        base_tokenizer::{TokenIdsWithOffsets, TruncationStrategy},
        constants::{
            ACCENT_MARKERS, ADDITIONAL_WHITESPACE_CHARS, CONTROL_CHARS,
            PUNCTUATION_CHARS, WHITESPACE_CHARS,
        },
    },
    vocab::Vocab,
    Mask, Offset, OffsetSize, Token, TokenRef,
};
use alloc::{borrow::ToOwned, string::String, vec::Vec};
use core::{borrow::BorrowMut, char, char::REPLACEMENT_CHARACTER, cmp::min};
use unicode_normalization::char::decompose_canonical;

pub type BpeCache = RwLock<BTreeMap<String, (Vec<String>, Vec<usize>)>>;

/// Cleans text by removing control characters and normalizing whitespace
pub fn clean_text(token: &mut Token, strict: bool) {
    let capacity = token.text.capacity();
    let mut cleaned_string = String::with_capacity(capacity);
    let mut character_mapping: Vec<OffsetSize> = Vec::with_capacity(capacity);
    for (character, position) in
        token.text.chars().zip(token.reference_offsets.iter())
    {
        if is_control(&character, strict)
            || character == '\x00'
            || character == REPLACEMENT_CHARACTER
        {
            continue;
        }
        if is_whitespace(&character) {
            cleaned_string.push(' ');
        } else {
            cleaned_string.push(character);
        }
        character_mapping.push(*position);
    }
    token.text = cleaned_string;
    token.reference_offsets = character_mapping;
    token.offset.begin = *token.reference_offsets.first().unwrap_or(&(0));
    token.offset.end = *token.reference_offsets.last().unwrap_or(&(0)) + 1;
}

/// Split a text on special tokens (like BOS/EOS/UNK markers), depending on the
/// vocabulary
pub fn split_on_special_tokens<'a>(
    token: TokenRef<'a>,
    vocab: &impl Vocab,
) -> Vec<TokenRef<'a>> {
    let test_substr = |s: &str| {
        for special_value in vocab.special_values().keys() {
            if s.starts_with(special_value.as_str()) {
                return (
                    special_value.len(),
                    special_value.chars().count(),
                    if BertVocab::UNKNOWN == special_value.as_str() {
                        Mask::Unknown
                    } else {
                        Mask::Special
                    },
                );
            }
        }
        (0, 0, Mask::None)
    };
    split_on_substr(token, test_substr, true)
}

/// Tokenizes CJK characters, each character will be a token
pub fn tokenize_cjk_chars(token: TokenRef) -> Vec<TokenRef> {
    split_on_char(token, is_cjk_char, true, Mask::CJK)
}

fn is_cjk_char(character: &char) -> bool {
    let u32_char = *character as u32;
    (0x4E00..=0x9FFF).contains(&u32_char)
        | (0x3400..=0x4DBF).contains(&u32_char)
        | (0x20000..=0x2A6DF).contains(&u32_char)
        | (0x2A700..=0x2B73F).contains(&u32_char)
        | (0x2B740..=0x2B81F).contains(&u32_char)
        | (0x2B820..=0x2CEAF).contains(&u32_char)
        | (0xF900..=0xFAFF).contains(&u32_char)
        | (0x2F800..=0x2FA1F).contains(&u32_char)
}

pub fn is_whitespace(character: &char) -> bool {
    WHITESPACE_CHARS.contains(&(*character as u32))
}

///    This is a custom method to check if a character is a control character.
/// The BERT tokenizer is taking any character whose unicode category starts
/// with `C` as a control character, which includes the traditional control `Cc`
/// category, but also the format `Cc`, private use `Co` and surrogate `Cs`. The
/// unassigned unicode category `Cn` has been skipped in order to avoid
/// unnecessary checks.    A faster method may be called by setting strict to
/// false and only check against the core control characters. To match the
/// original BERT tokenization, this should remain true.
pub fn is_control(character: &char, strict: bool) -> bool {
    if ADDITIONAL_WHITESPACE_CHARS.contains(character) {
        false
    } else if strict {
        let u32_char = *character as u32;
        (u32_char <= 0x001F)
            | (0x0080..=0x009F).contains(&u32_char)
            | (0xE0020..=0xE007F).contains(&u32_char)
            | (0xE000..=0xF8FF).contains(&u32_char)
            | (0xF0000..=0xFFFFD).contains(&u32_char)
            | (0x100000..=0x10FFFD).contains(&u32_char)
            | (0xD800..=0xDB7F).contains(&u32_char)
            | (0xDB80..=0xDBFF).contains(&u32_char)
            | (0xDC00..=0xDFFF).contains(&u32_char)
            | CONTROL_CHARS.contains(&u32_char)
    } else {
        character.is_control()
    }
}

pub fn is_punctuation(character: &char) -> bool {
    let u32_char = *character as u32;
    if (33..=47).contains(&u32_char)
        | (58..=64).contains(&u32_char)
        | (91..=96).contains(&u32_char)
        | (123..=126).contains(&u32_char)
    {
        true
    } else {
        PUNCTUATION_CHARS.contains(&u32_char)
    }
}

/// Simple tokenization based on whitespace only
pub fn whitespace_tokenize(token: TokenRef) -> Vec<TokenRef> {
    split_on_char(token, is_whitespace, false, Mask::Whitespace)
}

/// Remove diacritics
pub fn lowercase(token: &mut Token) {
    let capacity = token.text.capacity();
    let mut lower_cased_string: String = String::with_capacity(capacity);
    let mut character_mapping: Vec<OffsetSize> = Vec::with_capacity(capacity);
    for (character, position) in
        token.text.chars().zip(token.reference_offsets.iter())
    {
        for c in character.to_lowercase() {
            lower_cased_string.push(c);
            character_mapping.push(*position);
        }
    }
    token.text = lower_cased_string;
    token.reference_offsets = character_mapping;
    token.offset.begin = *token.reference_offsets.first().unwrap_or(&(0));
    token.offset.end = *token.reference_offsets.last().unwrap_or(&(0)) + 1;
}

/// Remove diacritics
pub fn strip_accents(token: &mut Token) {
    let capacity = token.text.capacity();
    let mut decomposed_string: String = String::with_capacity(capacity);
    let mut character_mapping: Vec<OffsetSize> = Vec::with_capacity(capacity);
    for (character, position) in
        token.text.chars().zip(token.reference_offsets.iter())
    {
        decompose_canonical(character, |c| {
            if !ACCENT_MARKERS.contains(&(c as u32)) {
                decomposed_string.push(c);
                character_mapping.push(*position);
            }
        });
    }
    token.text = decomposed_string;
    token.reference_offsets = character_mapping;
    token.offset.begin = *token.reference_offsets.first().unwrap_or(&(0));
    token.offset.end = *token.reference_offsets.last().unwrap_or(&(0)) + 1;
}

/// Split a token on punctuation
pub fn split_on_punct(token: TokenRef) -> Vec<TokenRef> {
    split_on_char(token, is_punctuation, true, Mask::Punctuation)
}

/// Split a token on one or more characters (given a character test function)
/// * token: The token to split
/// * test_character: A function that borrows a `char` and returns a boolean. If
///   true, a split will be made here
/// * add_separators: Add the separating characters to the tokens as well?
///   (bool), separating tokens will be indicated in the returned mask by the
///   value set in `set_mask`
pub fn split_on_char<'a, F>(
    token: TokenRef<'a>,
    test_character: F,
    add_separators: bool,
    set_mask: Mask,
) -> Vec<TokenRef<'a>>
where
    F: Fn(&char) -> bool,
{
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut charbegin: usize = 0;
    let mut bytesbegin: usize = 0;
    let mut charcount: usize = 0;

    if token.mask == Mask::None {
        // iterate over all characters, returning the byte position with each
        for (char_idx, (bytes_idx, c)) in token.text.char_indices().enumerate()
        {
            charcount += 1;
            if test_character(&c) {
                if charbegin < char_idx {
                    // add previous token
                    tokens.push(TokenRef {
                        text: &token.text
                            [bytesbegin..bytesbegin + (bytes_idx - bytesbegin)],
                        offset: Offset {
                            begin: token.offset.begin + charbegin as OffsetSize,
                            end: token.offset.begin + char_idx as OffsetSize,
                        },
                        reference_offsets: &token.reference_offsets
                            [charbegin..char_idx],
                        mask: Mask::None,
                    });
                }
                if add_separators {
                    // add separator as a singleton token
                    tokens.push(TokenRef {
                        text: &token.text[bytes_idx..bytes_idx + c.len_utf8()],
                        offset: Offset {
                            begin: token.offset.begin + char_idx as OffsetSize,
                            end: token.offset.begin
                                + char_idx as OffsetSize
                                + 1,
                        },
                        reference_offsets: &token.reference_offsets
                            [char_idx..char_idx + 1],
                        mask: set_mask,
                    });
                }
                // reset
                charbegin = char_idx + 1;
                bytesbegin = bytes_idx + c.len_utf8();
            }
        }
    }
    if charcount == 0 {
        // nothing done, return token as is
        tokens.push(token);
    } else if bytesbegin < token.text.len() {
        // add last buffered token if there is anything left
        if charcount == 0 {
            charcount = token.text.chars().count();
        }
        let bytes_idx = token.text.len();
        tokens.push(TokenRef {
            text: &token.text
                [bytesbegin..bytesbegin + (bytes_idx - bytesbegin)],
            offset: Offset {
                begin: token.offset.begin + charbegin as OffsetSize,
                end: token.offset.begin + charcount as OffsetSize,
            },
            reference_offsets: &token.reference_offsets[charbegin..charcount],
            mask: Mask::None,
        });
    }
    tokens
}

/// Split a token on one or more substrings (given a substring test function)
/// * token: The token to split
/// * test_str: A function that contains the string buffer from the current
///   point forward and
/// returns a 3-tuple with the length of the match in bytes, chars and the mask
/// to set (if the length is zero then there is no match.
/// * add_separators: Add the separating characters to the tokens as well?
///   (bool), separating tokens
/// will be indicated in the returned mask by the value set in `set_mask`, which
/// is returned by the test_substr function
pub fn split_on_substr<'a, F>(
    token: TokenRef<'a>,
    test_substr: F,
    add_separators: bool,
) -> Vec<TokenRef<'a>>
where
    F: Fn(&'a str) -> (usize, usize, Mask),
{
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut char_begin: usize = 0;
    let mut bytes_begin: usize = 0;
    let mut char_count: usize = 0;

    if token.mask == Mask::None {
        // don't process a token that already got marked in the mask
        // iterate over all characters, returning the byte position with each
        for (char_idx, (bytes_idx, _)) in token.text.char_indices().enumerate()
        {
            char_count += 1;
            let (matched_bytes, matched_chars, set_mask): (usize, usize, Mask) =
                test_substr(&token.text[bytes_idx..]);
            if matched_chars > 0 {
                if char_begin < char_idx {
                    // add previous token
                    let trimmed_text = token.text
                        [bytes_begin..bytes_begin + (bytes_idx - bytes_begin)]
                        .trim_end();
                    let trimmed_text_len = trimmed_text.chars().count();
                    if trimmed_text_len > 0 {
                        tokens.push(TokenRef {
                            text: trimmed_text,
                            offset: Offset {
                                begin: token.offset.begin
                                    + char_begin as OffsetSize,
                                end: token.offset.begin
                                    + (char_begin + trimmed_text_len)
                                        as OffsetSize,
                            },
                            reference_offsets: &token.reference_offsets
                                [char_begin..(char_begin + trimmed_text_len)],
                            mask: Mask::None,
                        });
                    }
                }
                if add_separators {
                    // add separator as a singleton token
                    tokens.push(TokenRef {
                        text: &token.text[bytes_idx..bytes_idx + matched_bytes],
                        offset: Offset {
                            begin: token.offset.begin + char_idx as OffsetSize,
                            end: token.offset.begin
                                + (char_idx + matched_chars) as OffsetSize,
                        },
                        reference_offsets: &token.reference_offsets
                            [char_idx..(char_idx + matched_chars)],
                        mask: set_mask,
                    });
                }
                // reset
                char_begin = char_idx + matched_chars;
                bytes_begin = bytes_idx + matched_bytes;
            }
        }
    }
    if bytes_begin < token.text.len() {
        // add last buffered token if there is anything left
        let bytes_idx = token.text.len();
        let text =
            &token.text[bytes_begin..bytes_begin + (bytes_idx - bytes_begin)];
        if char_count == 0 {
            char_count = text.chars().count();
        }
        tokens.push(TokenRef {
            text,
            offset: Offset {
                begin: token.offset.begin + char_begin as OffsetSize,
                end: token.offset.begin + char_count as OffsetSize,
            },
            reference_offsets: &token.reference_offsets[char_begin..char_count],
            mask: Mask::None,
        });
    }
    tokens
}

/// Tokenize a token into word pieces according to the supplied vocabulary
/// Continuation word pieces will all have the suffix `##`
pub fn tokenize_wordpiece(
    token: TokenRef,
    vocab: &impl Vocab,
    max_word_len: usize,
) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    if token.text.chars().count() > max_word_len {
        tokens.push(Token {
            text: BertVocab::UNKNOWN.to_owned(),
            offset: token.offset,
            reference_offsets: token.reference_offsets.to_vec(),
            mask: Mask::Unknown,
        });
    } else {
        let char_indices: Vec<usize> =
            token.text.char_indices().map(|v| v.0).collect();
        let max_end: usize = char_indices.last().unwrap()
            + token.text.chars().last().unwrap().len_utf8();
        let mut start: usize = 0; // bytes
        let mut pos_begin = 0; // chars
        let mut pos_end;
        let mut end;
        while start < max_end {
            // bytes
            end = max_end;
            pos_end = char_indices.len(); // chars
            let mut is_unk: bool = true; // out of vocabulary? to be falsified
            while start < end {
                let mut substr = token.text[start..end].to_owned();
                let char_length = substr.chars().count();
                let sub_offset = Offset {
                    begin: token.offset.begin + pos_begin as OffsetSize,
                    end: token.offset.begin
                        + pos_begin as OffsetSize
                        + char_length as OffsetSize,
                };
                if start > 0 {
                    substr = format!("##{}", substr);
                }
                if vocab.values().contains_key(&substr) {
                    tokens.push(Token {
                        text: substr,
                        offset: sub_offset,
                        reference_offsets: token.reference_offsets
                            [pos_begin..(pos_begin + char_length)]
                            .to_vec(),
                        mask: if start > 0 {
                            Mask::Continuation
                        } else {
                            token.mask
                        },
                    });
                    is_unk = false;
                    break;
                }
                pos_end -= 1;
                end = char_indices[pos_end];
            }
            if is_unk {
                return vec![Token {
                    text: BertVocab::UNKNOWN.to_owned(),
                    offset: token.offset,
                    reference_offsets: token.reference_offsets.to_vec(),
                    mask: Mask::Unknown,
                }];
            }
            start = end;
            pos_begin = pos_end;
        }

        // fix the mask, set Mask::Begin where a sequence of continuations is
        // introduced
        fix_mask(&mut tokens);
    }

    tokens
}

/// # Truncates a sequence pair in place to the maximum length.
///
///   * tokens_1: list of tokenized input ids. Can be obtained from a string by
///     chaining the `tokenize` and `convert_tokens_to_ids` methods.
///   * tokens_2: Optional second list of input ids. Can be obtained from a
///     string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
///   * offsets: list of offsets for tokens_1 (must be same length or empty if
///     not used at all)
///   * offsets_2: optional second list of offsets for tokens_2 (must be same
///     length or empty if not used at all)
///   * tokens_2: Optional second list of input ids. Can be obtained from a
///     string by chaining the `tokenize` and `convert_tokens_to_ids` methods.
///   * num_tokens_to_remove number of tokens to remove using the truncation
///     strategy
///   * truncation_strategy: truncation strategy
///       - TruncationStrategy::LongestFirst (default) Iteratively reduce the
///         inputs sequence until the input is under max_length starting from
///         the longest one at each token (when there is a pair of input
///         sequences). Overflowing tokens only contains overflow from the first
///         sequence.
///       - TruncationStrategy::OnlyFirst: Only truncate the first sequence.
///         raise an error if the first sequence is shorter or equal to than
///         num_tokens_to_remove.
///       - TruncationStrategy::OnlySecond: Only truncate the second sequence
///       - TruncationStrategy::DoNotTruncate: Does not truncate (raise an error
///         if the input sequence is longer than max_length)
///   * stride If set to a number along with max_length, the overflowing tokens
///     returned will contain some tokens from the main sequence returned. The
///     value of this argument defines the number of additional tokens.
pub fn truncate_sequences(
    mut token_ids_with_offsets_1: TokenIdsWithOffsets,
    mut token_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    num_tokens_to_remove: usize,
    truncation_strategy: &TruncationStrategy,
    stride: usize,
) -> Result<
    (
        TokenIdsWithOffsets,
        Option<TokenIdsWithOffsets>,
        Vec<i64>,
        Vec<Option<Offset>>,
    ),
    &str,
> {
    if num_tokens_to_remove == 0 {
        Ok((
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            Vec::new(),
            Vec::new(),
        ))
    } else if let Some(token_ids_with_offsets_2_value) =
        token_ids_with_offsets_2.borrow_mut()
    {
        match truncation_strategy {
            TruncationStrategy::LongestFirst => {
                if (token_ids_with_offsets_1.ids.len()
                    + token_ids_with_offsets_2_value.ids.len())
                    >= num_tokens_to_remove
                {
                    let mut overflow_tokens: Vec<i64> =
                        Vec::with_capacity(num_tokens_to_remove + stride);
                    let mut overflow_offsets: Vec<Option<Offset>> =
                        Vec::with_capacity(num_tokens_to_remove + stride);
                    for _ in 0..num_tokens_to_remove {
                        if token_ids_with_offsets_1.ids.len()
                            >= token_ids_with_offsets_2_value.ids.len()
                        {
                            overflow_tokens.insert(
                                0,
                                token_ids_with_offsets_1.ids.pop().unwrap(),
                            );
                            if !token_ids_with_offsets_1.offsets.is_empty() {
                                overflow_offsets.insert(
                                    0,
                                    token_ids_with_offsets_1
                                        .offsets
                                        .pop()
                                        .unwrap(),
                                );
                            }
                            token_ids_with_offsets_1.reference_offsets.pop();
                            if !token_ids_with_offsets_1.masks.is_empty() {
                                token_ids_with_offsets_1.masks.pop();
                            }
                        } else {
                            overflow_tokens.insert(
                                0,
                                token_ids_with_offsets_2_value
                                    .ids
                                    .pop()
                                    .unwrap(),
                            );
                            if !token_ids_with_offsets_2_value
                                .offsets
                                .is_empty()
                            {
                                overflow_offsets.insert(
                                    0,
                                    token_ids_with_offsets_2_value
                                        .offsets
                                        .pop()
                                        .unwrap(),
                                );
                            }
                            token_ids_with_offsets_2_value
                                .reference_offsets
                                .pop();
                            if !token_ids_with_offsets_2_value.masks.is_empty()
                            {
                                token_ids_with_offsets_2_value.masks.pop();
                            }
                        }
                    }
                    let window_len =
                        min(token_ids_with_offsets_1.ids.len(), stride);
                    if window_len > 0 {
                        let slice: &[i64] = &token_ids_with_offsets_1.ids
                            [token_ids_with_offsets_1.ids.len() - window_len..];
                        overflow_tokens.splice(0..0, slice.iter().cloned());
                        if !token_ids_with_offsets_1.offsets.is_empty() {
                            let offset_slice: &[Option<Offset>] =
                                &token_ids_with_offsets_1.offsets
                                    [token_ids_with_offsets_1.offsets.len()
                                        - window_len..];
                            overflow_offsets
                                .splice(0..0, offset_slice.iter().cloned());
                        }
                    }
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err("Combined sequence length too short for requested truncation amount")
                }
            },
            TruncationStrategy::OnlyFirst => {
                if token_ids_with_offsets_1.ids.len() >= num_tokens_to_remove {
                    let (overflow_tokens, overflow_offsets) =
                        truncate_with_overflow(
                            &mut token_ids_with_offsets_1.ids,
                            token_ids_with_offsets_1.offsets.as_mut(),
                            token_ids_with_offsets_1.reference_offsets.as_mut(),
                            token_ids_with_offsets_1.masks.as_mut(),
                            num_tokens_to_remove,
                            stride,
                        );
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err("First sequence too short for first only truncation")
                }
            },
            TruncationStrategy::OnlySecond => {
                if token_ids_with_offsets_2_value.ids.len()
                    >= num_tokens_to_remove
                {
                    let (overflow_tokens, overflow_offsets) =
                        truncate_with_overflow(
                            &mut token_ids_with_offsets_2_value.ids,
                            token_ids_with_offsets_2_value.offsets.as_mut(),
                            token_ids_with_offsets_2_value
                                .reference_offsets
                                .as_mut(),
                            token_ids_with_offsets_2_value.masks.as_mut(),
                            num_tokens_to_remove,
                            stride,
                        );
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err("Second sequence too short for second only truncation")
                }
            },
            TruncationStrategy::DoNotTruncate => {
                Err("Truncation needed but no truncation requested")
            },
        }
    } else if token_ids_with_offsets_1.ids.len() >= num_tokens_to_remove {
        match truncation_strategy {
            TruncationStrategy::LongestFirst
            | TruncationStrategy::OnlyFirst => {
                let (overflow_tokens, overflow_offsets) =
                    truncate_with_overflow(
                        &mut token_ids_with_offsets_1.ids,
                        &mut token_ids_with_offsets_1.offsets,
                        &mut token_ids_with_offsets_1.reference_offsets,
                        &mut token_ids_with_offsets_1.masks,
                        num_tokens_to_remove,
                        stride,
                    );
                Ok((
                    token_ids_with_offsets_1,
                    token_ids_with_offsets_2,
                    overflow_tokens,
                    overflow_offsets,
                ))
            },
            TruncationStrategy::OnlySecond => Err(
                "Invalid truncation strategy for single sentence truncation",
            ),
            TruncationStrategy::DoNotTruncate => {
                Err("Truncation needed but no truncation requested")
            },
        }
    } else {
        Err("First sequence too short for first only truncation")
    }
}

fn truncate_with_overflow(
    sequence: &mut Vec<i64>,
    offsets: &mut Vec<Option<Offset>>,
    original_positions: &mut Vec<Vec<OffsetSize>>,
    mask: &mut Vec<Mask>,
    num_tokens_to_remove: usize,
    stride: usize,
) -> (Vec<i64>, Vec<Option<Offset>>) {
    if !offsets.is_empty() {
        assert_eq!(sequence.len(), offsets.len());
    }
    if !mask.is_empty() {
        assert_eq!(sequence.len(), mask.len());
    }
    let cutoff = sequence.len() - num_tokens_to_remove;
    let mut overflow_tokens = sequence.split_off(cutoff);
    let mut overflow_offsets = if !offsets.is_empty() {
        offsets.split_off(cutoff)
    } else {
        Vec::new()
    };
    if !mask.is_empty() {
        mask.truncate(cutoff);
        original_positions.truncate(cutoff);
    }
    let window_len = min(sequence.len(), stride);
    if window_len > 0 {
        let slice: &[i64] = &sequence[sequence.len() - window_len..];
        overflow_tokens.splice(0..0, slice.iter().cloned());
        if !offsets.is_empty() {
            let offset_slice: &[Option<Offset>] =
                &offsets[offsets.len() - window_len..];
            overflow_offsets.splice(0..0, offset_slice.iter().cloned());
        }
    }
    (overflow_tokens, overflow_offsets)
}

pub fn fix_mask(tokens: &mut Vec<Token>) {
    for i in 1..tokens.len() {
        if tokens[i].mask == Mask::Continuation
            && tokens[i - 1].mask == Mask::None
        {
            if let Some(token) = tokens.get_mut(i - 1) {
                token.mask = Mask::Begin;
            }
        }
    }
}

///Default bpe function, as called by Roberta and GPT2
pub fn bpe(token: &str, bpe_ranks: &BpePairVocab) -> (Vec<String>, Vec<usize>) {
    let sub_tokens = token
        .chars()
        .map(|v| v.to_string())
        .collect::<Vec<String>>();

    let mut output = (sub_tokens, false);
    loop {
        output = group_common_pairs(output.0, bpe_ranks);
        if output.1 {
            break;
        }
    }
    let char_counts = output.0.iter().map(|v| v.chars().count()).collect();
    (output.0, char_counts)
}

pub fn split_on_bpe_pairs<'a, F>(
    token: TokenRef<'a>,
    bpe_function: F,
    bpe_ranks: &BpePairVocab,
    cache: &BpeCache,
    as_bytes: bool,
) -> Vec<Token>
where
    F: Fn(&str, &BpePairVocab) -> (Vec<String>, Vec<usize>),
{
    let mut tokens: Vec<Token> = Vec::new();
    let text: String;
    let reference_offsets_placeholder: Vec<OffsetSize>;
    let (text, reference_offsets) = if as_bytes {
        reference_offsets_placeholder = bytes_offsets(token.text)
            .iter()
            .map(|&pos| token.reference_offsets[pos])
            .collect();
        text = token
            .text
            .as_bytes()
            .iter()
            .map(|v| BYTES_TO_UNICODE.get(v).unwrap())
            .collect();
        (text.as_str(), reference_offsets_placeholder.as_slice())
    } else {
        (token.text, token.reference_offsets)
    };

    let cached: bool = if let Ok(ref mut cache) = cache.try_read() {
        match cache.get(text) {
            Some((cached_tokens, char_counts)) => {
                let mut start = 0;
                for (idx, (sub_token, &char_count)) in
                    cached_tokens.iter().zip(char_counts.iter()).enumerate()
                {
                    tokens.push(Token {
                        text: sub_token.clone(),
                        offset: Offset {
                            begin: reference_offsets[start],
                            end: reference_offsets[start + char_count - 1] + 1,
                        },
                        reference_offsets: reference_offsets
                            [start as usize..start as usize + char_count]
                            .to_vec(),
                        mask: {
                            if cached_tokens.len() > 1 {
                                if idx == 0 {
                                    Mask::Begin
                                } else {
                                    Mask::Continuation
                                }
                            } else {
                                Mask::None
                            }
                        },
                    });
                    start += char_count;
                }
                true
            },
            None => false,
        }
    } else {
        false
    };

    if !cached {
        let (bpe_output, char_counts) = bpe_function(text, bpe_ranks);
        if let Ok(mut cache) = cache.try_write() {
            cache.insert(
                text.to_owned(),
                (bpe_output.clone(), char_counts.clone()),
            );
        }
        let mut start = 0;
        for (idx, (sub_token, &char_count)) in
            bpe_output.iter().zip(char_counts.iter()).enumerate()
        {
            tokens.push(Token {
                text: sub_token.clone(),
                offset: Offset {
                    begin: reference_offsets[start],
                    end: reference_offsets[start + char_count - 1] + 1,
                },
                reference_offsets: reference_offsets
                    [start as usize..start as usize + char_count]
                    .to_vec(),
                mask: {
                    if bpe_output.len() > 1 {
                        if idx == 0 {
                            Mask::Begin
                        } else {
                            Mask::Continuation
                        }
                    } else {
                        Mask::None
                    }
                },
            });
            start += char_count;
        }
    }
    tokens
}

pub fn split_on_regex_with_lookahead<'a>(
    token: TokenRef<'a>,
    pattern_lookahead: &Regex,
    pattern_tokenization: &Regex,
) -> Vec<TokenRef<'a>> {
    if token.mask == Mask::None {
        let mut sub_words: Vec<&str> = vec![];
        let mut splits: Vec<&str> = vec![];

        let mut i: usize = 0;
        let mut end_byte: usize;
        for hit in pattern_lookahead.find_iter(token.text) {
            let mut hit_chars = hit.as_str().chars().rev();
            let start = hit_chars.next().unwrap();
            let sep = hit_chars.next().unwrap();
            end_byte = hit.end() - sep.len_utf8() - start.len_utf8();
            splits.push(&token.text[i..end_byte]);
            i = end_byte;
        }
        splits.push(&token.text[i..]);

        for sub_word in splits {
            for hit in pattern_tokenization.find_iter(sub_word) {
                sub_words.push(hit.as_str());
            }
        }

        let mut output_tokens: Vec<TokenRef> =
            Vec::with_capacity(sub_words.len());
        let mut begin_char: usize = 0;
        let mut end_char: usize;
        for sub_word in sub_words {
            end_char = begin_char + sub_word.chars().count();
            output_tokens.push(TokenRef {
                text: sub_word,
                offset: Offset::new(
                    token.offset.begin + begin_char as OffsetSize,
                    token.offset.begin + end_char as OffsetSize,
                ),
                reference_offsets: &token.reference_offsets
                    [begin_char..end_char],
                mask: Default::default(),
            });
            begin_char = end_char;
        }

        output_tokens
    } else {
        vec![token]
    }
}
