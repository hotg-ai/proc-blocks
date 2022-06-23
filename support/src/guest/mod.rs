//! Types and utilities for implementing a proc-block.

#[macro_use]
mod macros;

pub(crate) mod bindings;
mod element_type;
mod errors;
mod logging;
mod metadata;
pub mod parse;
mod proc_block;
mod tensor;

use std::{panic::PanicInfo, sync::Once};

pub use self::{
    bindings::{
        abort, Argument, ArgumentError, ArgumentErrorReason, ArgumentHint,
        ArgumentMetadata, ArgumentType, CreateError, Dimensions, ElementType,
        ElementTypeConstraint, InvalidInput, InvalidInputReason, MediaType,
        Metadata, RunError, Tensor, TensorConstraint, TensorConstraints,
        TensorHint, TensorMetadata,
    },
    element_type::{PrimitiveTensorElement, UnknownElementType},
    proc_block::ProcBlock,
};

getrandom::register_custom_getrandom!(host_rng);

fn host_rng(buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    bindings::get_random(buffer);
    Ok(())
}

/// Run any necessary initialization code.
pub(crate) fn ensure_initialized() {
    static ONCE: Once = Once::new();

    ONCE.call_once(|| {
        let _ = logging::initialize_logger();
        std::panic::set_hook(Box::new(panic_hook));
    });
}

fn panic_hook(panic_info: &PanicInfo<'_>) {
    let location = panic_info.location();
    let file = location.map(|loc| loc.file());
    let line = location.map(|loc| loc.line());

    tracing::error!(panic.file = file, panic.line = line, "{panic_info}");
}
