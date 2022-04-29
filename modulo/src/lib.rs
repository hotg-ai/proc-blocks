#![allow(dead_code)]

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

use std::fmt::Display;

use crate::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError,
    },
    runtime_v1::{
        ArgumentMetadata, DimensionsParam, ElementType, GraphContext,
        KernelContext, Metadata, TensorMetadata, TensorParam, TensorResult,
    },
};
use hotg_rune_proc_blocks::BufferExt;
use num_traits::{FromPrimitive, ToPrimitive};

pub struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));

        let modulo = ArgumentMetadata::new("modulo");
        modulo.add_hint(&runtime_v1::non_negative_number());
        metadata.add_argument(&modulo);
        let element_type = ArgumentMetadata::new("element_type");
        element_type
            .set_description("The type of tensor this proc-block will accept");
        element_type.set_default_value("f64");
        element_type.add_hint(&runtime_v1::interpret_as_string_in_enum(&[
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        ]));
        metadata.add_argument(&element_type);

        let input = TensorMetadata::new("input");
        metadata.add_input(&input);

        let output = TensorMetadata::new("output");
        metadata.add_output(&output);

        runtime_v1::register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id).ok_or_else(|| {
            GraphError::Other("Unable to load the graph context".to_string())
        })?;

        // make sure the modulus is valid
        let _ = get_modulus(|n| ctx.get_argument(n))
            .map_err(GraphError::InvalidArgument)?;

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("u8") => ElementType::U8,
            Some("i8") => ElementType::I8,
            Some("u16") => ElementType::U16,
            Some("i16") => ElementType::I16,
            Some("u32") => ElementType::U32,
            Some("i32") => ElementType::I32,
            Some("f32") => ElementType::F32,
            Some("u64") => ElementType::U64,
            Some("i64") => ElementType::I64,
            Some("f64") | None => ElementType::F64,
            Some(_) => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::InvalidValue(
                        "Unsupported element type".to_string(),
                    ),
                }))
            },
        };

        ctx.add_input_tensor("input", element_type, DimensionsParam::Dynamic);
        ctx.add_output_tensor("output", element_type, DimensionsParam::Dynamic);

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id).ok_or_else(|| {
            KernelError::Other("Unable to load the kernel context".to_string())
        })?;

        let modulus = get_modulus(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let TensorResult {
            dimensions,
            element_type,
            mut buffer,
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "input".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        // Note: The "element_type" argument is only used while constructing the
        // ML pipeline. We see its effect at runtime in the form of the tensor
        // data variant that gets used.

        match element_type {
            ElementType::U8 => {
                modulus_in_place(buffer.elements_mut::<u8>(), modulus)?
            },
            ElementType::I8 => {
                modulus_in_place(buffer.elements_mut::<i8>(), modulus)?
            },
            ElementType::U16 => {
                modulus_in_place(buffer.elements_mut::<u16>(), modulus)?
            },
            ElementType::I16 => {
                modulus_in_place(buffer.elements_mut::<i16>(), modulus)?
            },
            ElementType::U32 => {
                modulus_in_place(buffer.elements_mut::<u32>(), modulus)?
            },
            ElementType::I32 => {
                modulus_in_place(buffer.elements_mut::<i32>(), modulus)?
            },
            ElementType::F32 => {
                modulus_in_place(buffer.elements_mut::<f32>(), modulus)?
            },
            ElementType::U64 => {
                modulus_in_place(buffer.elements_mut::<u64>(), modulus)?
            },
            ElementType::I64 => {
                modulus_in_place(buffer.elements_mut::<i64>(), modulus)?
            },
            ElementType::F64 => {
                modulus_in_place(buffer.elements_mut::<f64>(), modulus)?
            },
            ElementType::Utf8 => {
                return Err(KernelError::Other(
                    "String tensors aren't supported".to_string(),
                ))
            },
        }

        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type,
                dimensions: &dimensions,
                buffer: &buffer,
            },
        );

        Ok(())
    }
}

fn modulus_in_place<T>(
    values: &mut [T],
    modulus: f64,
) -> Result<(), KernelError>
where
    T: ToPrimitive + FromPrimitive + Copy + Display,
{
    for value in values {
        let as_float = value.to_f64().ok_or_else(|| error(*value))?;
        let after_modulus = as_float % modulus;
        *value = T::from_f64(after_modulus).ok_or_else(|| error(*value))?;
    }

    Ok(())
}

fn error(value: impl Display) -> KernelError {
    KernelError::Other(format!(
        "Unable to convert `{}` to/from a double",
        value
    ))
}

fn get_modulus(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<f64, InvalidArgument> {
    let value = match get_argument("modulus") {
        Some(s) => s,
        None => {
            return Err(InvalidArgument {
                name: "modulus".to_string(),
                reason: BadArgumentReason::NotFound,
            })
        },
    };

    let value = value.parse::<f64>().map_err(|e| InvalidArgument {
        name: "modulus".to_string(),
        reason: BadArgumentReason::InvalidValue(e.to_string()),
    })?;

    if value > 0.0 {
        Ok(value)
    } else {
        Err(InvalidArgument {
            name: "modulus".to_string(),
            reason: BadArgumentReason::InvalidValue(
                "The modulus must be a positive, non-zero number".to_string(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_modulus() {
        let mut values = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];

        modulus_in_place(&mut values, 2.0).unwrap();

        assert_eq!(values, [0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
