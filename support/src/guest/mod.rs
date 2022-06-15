//! Types and utilities for implementing a proc-block.

#[macro_use]
mod macros;

pub(crate) mod bindings;
mod element_type;
mod errors;
mod helpers;
mod logging;
mod metadata;
mod proc_block;
mod tensor;

pub use self::{
    bindings::{
        abort, Argument, ArgumentError, ArgumentErrorReason, ArgumentHint,
        ArgumentMetadata, ArgumentType, CreateError, Dimensions, ElementType,
        ElementTypeConstraint, InvalidInput, InvalidInputReason, MediaType,
        Metadata, RunError, Tensor, TensorConstraint, TensorConstraints,
        TensorHint, TensorMetadata,
    },
    element_type::PrimitiveTensorElement,
    helpers::parse_arg,
    proc_block::ProcBlock,
};

getrandom::register_custom_getrandom!(host_rng);

fn host_rng(buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    bindings::get_random(buffer);
    Ok(())
}
