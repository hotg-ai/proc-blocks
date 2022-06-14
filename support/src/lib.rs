#![doc = include_str!("../README.md")]

pub extern crate ndarray;

#[cfg(feature = "runtime_v1")]
mod bindings;

mod buffer_ext;
pub mod common;
mod string_builder;
mod value_type;

use std::sync::Mutex;

pub use crate::{
    buffer_ext::BufferExt,
    string_builder::{string_tensor_from_ndarray, StringBuilder},
    value_type::{SliceExt, ValueType},
};

#[cfg(feature = "runtime_v1")]
pub use bindings::runtime_v1;
use once_cell::sync::Lazy;
use rand::{prelude::SmallRng, Rng, SeedableRng};

pub mod prelude {
    #[cfg(feature = "runtime_v1")]
    pub use crate::bindings::{
        ContextErrorExt, ContextExt, InvalidArgumentExt,
    };
}

// Note: getrandom is pulled in by the linfa_logistic crate
getrandom::register_custom_getrandom!(unsupported_rng);

fn unsupported_rng(buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    // FIXME: We should probably seed this with something more useful.
    static RNG: Lazy<Mutex<SmallRng>> =
        Lazy::new(|| Mutex::new(SmallRng::from_seed([0; 16])));

    RNG.lock().unwrap().fill(buffer);
    Ok(())
}
