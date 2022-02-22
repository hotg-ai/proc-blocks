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

use anyhow::Result;
use std::{collections::BTreeMap, hash::Hash, string::String};

pub(crate) fn swap_key_values<
    T: Clone,
    U: Hash + Eq + Copy + core::cmp::Ord,
>(
    input_hashmap: &BTreeMap<T, U>,
) -> BTreeMap<U, T> {
    input_hashmap
        .iter()
        .map(|(key, &value)| (value, key.clone()))
        .collect()
}

#[derive(Debug, Clone)]
pub enum TokenError {
    TokenNotFound { word: String },
}

/// # Base Vocab trait
/// Defines a common interface to the vocabularies for use in the tokenizers.
pub trait Vocab {
    // Associative function returning the unknown value for the vocabulary
    // fn unknown_value() -> &'static str;

    // / Returns the unknown value on an instance
    // fn get_unknown_value(&self) -> &'static str;

    /// Return the map of token strings to IDs
    fn values(&self) -> &BTreeMap<String, i64>;

    // Return the map of token IDs to strings
    fn indices(&self) -> &BTreeMap<i64, String>;

    // Return the map of token strings to IDs
    fn special_values(&self) -> &BTreeMap<String, i64>;

    /// Return the map of token IDs to strings for special values
    fn special_indices(&self) -> &BTreeMap<i64, String>;

    /// Converts a token to an id, provided a `BTreeMap` of values, a `BTreeMap`
    /// of special values and the unknown value token string representation.
    /// This is not meant to be directly used, the method `token_to_id`
    /// offers a more convenient interface for most vocabularies, but needs to
    /// be implemented by the specific vocabulary.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    /// - values (`&BTreeMap<String, i64>`): mapping from tokens to ids
    /// - special_values (`&BTreeMap<String, i64>`): mapping from special tokens
    ///   to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `i64`: index value for the provided token
    fn _token_to_id(
        &self,
        token: &str,
        values: &BTreeMap<String, i64>,
        special_values: &BTreeMap<String, i64>,
        unknown_value: &str,
    ) -> i64 {
        match special_values.get(token) {
            Some(index) => *index,
            None => match values.get(token) {
                Some(index) => *index,
                None => *values.get(unknown_value).unwrap(),
            },
        }
    }

    /// Converts a token to an id.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    ///
    /// # Returns
    /// - `i64`: token index for the value provided. If not found in the
    ///   indices, returns the unknown token index
    fn token_to_id(&self, token: &str) -> i64;

    /// Register a token as a special value
    ///
    /// # Parameters
    /// - token (`&str`): token to register as a special value
    /// - values (`&BTreeMap<String, i64>`): mapping from tokens to ids. This
    ///   should contain the token to add and will be used to read the id for
    ///   registration in `special_values`
    /// - special_values (`&BTreeMap<String, i64>`): mapping from special tokens
    ///   to ids
    fn _register_as_special_value(
        token: &str,
        values: &BTreeMap<String, i64>,
        special_values: &mut BTreeMap<String, i64>,
    ) -> Result<(), TokenError> {
        let token_id = match values.get(token) {
            Some(index) => *index,
            None => {
                return Err(TokenError::TokenNotFound {
                    word: token.to_string(),
                });
            },
        };
        special_values.insert(String::from(token), token_id);
        Ok(())
    }

    /// Converts an id to a token, provided a `HashMap` of values, a `HashMap`
    /// of special values and the unknown value token string representation.
    /// This is not meant to be directly used, the method `id_to_token`
    /// offers a more convenient interface for most vocabularies, but needs to
    /// be implemented by the specific vocabulary.
    ///
    /// # Parameters
    /// - id (`&i64`): token id to convert
    /// - indices (`&HashMap<i64, String>`): mapping from tokens to ids
    /// - special_indices (`&HashMap<i64, String>`): mapping from special tokens
    ///   to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `String`: token value for the index provided. If not found in the
    ///   indices, returns the unknown token value

    fn _id_to_token<'a>(
        &self,
        id: i64,
        values: &'a BTreeMap<i64, String>,
        special_indices: &'a BTreeMap<i64, String>,
        unknown_value: &'a str,
    ) -> &'a str {
        special_indices
            .get(&id)
            .or_else(|| values.get(&id))
            .map(|s| s.as_str())
            .unwrap_or(unknown_value)
    }

    /// # Returns
    /// - `String`: token value for the index provided. If not found in the
    ///   indices, returns the unknown token value
    fn id_to_token(&self, id: i64) -> &str;
}
