#![doc = include_str!("../README.md")]

pub extern crate ndarray;

mod macros;
mod strings;

#[cfg(feature = "guest")]
pub mod guest;

use std::sync::Mutex;

pub use crate::strings::{decode_strings, StringBuilder};

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
