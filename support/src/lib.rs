#![doc = include_str!("../README.md")]

pub extern crate ndarray;

mod buffer_ext;
pub mod common;
mod macros;
mod string_builder;
mod value_type;

use std::sync::Mutex;

pub use crate::{
    buffer_ext::BufferExt,
    string_builder::{string_tensor_from_ndarray, StringBuilder},
    value_type::{SliceExt, ValueType},
};
use once_cell::sync::Lazy;
use rand::{prelude::SmallRng, Rng, SeedableRng};

// Note: getrandom is pulled in by the linfa_logistic crate
getrandom::register_custom_getrandom!(unsupported_rng);

fn unsupported_rng(buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    // FIXME: We should probably seed this with something more useful.
    static RNG: Lazy<Mutex<SmallRng>> =
        Lazy::new(|| Mutex::new(SmallRng::from_seed(Default::default())));

    RNG.lock().unwrap().fill(buffer);
    Ok(())
}
