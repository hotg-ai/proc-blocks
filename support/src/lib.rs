#![doc = include_str!("../README.md")]

pub extern crate ndarray;

mod buffer_ext;
pub mod common;
mod string_builder;
mod value_type;

pub use crate::{
    buffer_ext::BufferExt,
    string_builder::StringBuilder,
    value_type::{SliceExt, ValueType},
};
