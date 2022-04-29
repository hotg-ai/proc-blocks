#![doc = include_str!("../README.md")]

pub extern crate ndarray;

#[cfg(feature = "runtime_v1")]
mod bindings;

mod buffer_ext;
pub mod common;
mod string_builder;
mod value_type;

pub use crate::{
    buffer_ext::BufferExt,
    string_builder::{string_tensor_from_ndarray, StringBuilder},
    value_type::{SliceExt, ValueType},
};

#[cfg(feature = "runtime_v1")]
pub use bindings::runtime_v1;

pub mod prelude {
    #[cfg(feature = "runtime_v1")]
    pub use crate::bindings::{ContextErrorExt, InvalidArgumentExt};
}
