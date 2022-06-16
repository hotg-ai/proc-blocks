#![doc = include_str!("../README.md")]

pub extern crate ndarray;

mod macros;
mod strings;

#[cfg(feature = "guest")]
pub mod guest;

pub use crate::strings::{decode_strings, StringBuilder};
