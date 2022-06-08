#![allow(dead_code)]

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v2.wit");

use crate::proc_block_v2::*;

use hotg_rune_proc_blocks::{BufferExt, ValueType};
use num_traits::{FromPrimitive, ToPrimitive};
use wit_bindgen_rust::Handle;

hotg_rune_proc_blocks::generate_support!(crate::proc_block_v2);

pub struct ProcBlockV2;

impl proc_block_v2::ProcBlockV2 for ProcBlockV2 {
    fn metadata() -> Metadata {
        Metadata {
            name: "Modulo".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            tags: Vec::new(),
            description: Some(env!("CARGO_PKG_DESCRIPTION").to_string()),
            homepage: Some(env!("CARGO_PKG_HOMEPAGE").to_string()),
            repository: Some(env!("CARGO_PKG_REPOSITORY").to_string()),
            arguments: vec![ArgumentMetadata {
                name: "modulus".to_string(),
                description: Some("The modulus operand".to_string()),
                default_value: None,
                hints: vec![ArgumentHint::ArgumentType(ArgumentType::Float)],
            }],
            inputs: vec![TensorMetadata {
                name: "input".to_string(),
                description: None,
                hints: Vec::new(),
            }],
            outputs: vec![TensorMetadata {
                name: "output".to_string(),
                description: None,
                hints: Vec::new(),
            }],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    modulus: f64,
}

impl TryFrom<Vec<Argument>> for Node {
    type Error = ArgumentError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let modulus = support::parse_arg(&args, "modulus")?;

        Ok(Node { modulus })
    }
}

impl proc_block_v2::Node for Node {
    fn new(args: Vec<Argument>) -> Result<Handle<crate::Node>, ArgumentError> {
        Node::try_from(args).map(Handle::new)
    }

    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint {
                name: "input".to_string(),
                element_type: !ElementTypeConstraint::UTF8,
                dimensions: Dimensions::Dynamic,
            }],
            outputs: vec![TensorConstraint {
                name: "output".to_string(),
                element_type: !ElementTypeConstraint::UTF8,
                dimensions: Dimensions::Dynamic,
            }],
        }
    }

    fn run(
        &self,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, proc_block_v2::KernelError> {
        let mut tensor = inputs
            .into_iter()
            .find(|tensor| tensor.name == "input")
            .ok_or_else(|| {
                KernelError::InvalidInput(InvalidInput {
                    name: "input".to_string(),
                    reason: InvalidInputReason::NotFound,
                })
            })?;

        tensor.name = "output".to_string();

        match tensor.element_type {
            ElementType::U8 => {
                modulus_in_place::<u8>(&mut tensor.buffer, self.modulus)
            },
            ElementType::I8 => {
                modulus_in_place::<i8>(&mut tensor.buffer, self.modulus)
            },
            ElementType::U16 => {
                modulus_in_place::<u16>(&mut tensor.buffer, self.modulus)
            },
            ElementType::I16 => {
                modulus_in_place::<i16>(&mut tensor.buffer, self.modulus)
            },
            ElementType::U32 => {
                modulus_in_place::<u32>(&mut tensor.buffer, self.modulus)
            },
            ElementType::I32 => {
                modulus_in_place::<i32>(&mut tensor.buffer, self.modulus)
            },
            ElementType::F32 => {
                modulus_in_place::<f32>(&mut tensor.buffer, self.modulus)
            },
            ElementType::U64 => {
                modulus_in_place::<u64>(&mut tensor.buffer, self.modulus)
            },
            ElementType::I64 => {
                modulus_in_place::<i64>(&mut tensor.buffer, self.modulus)
            },
            ElementType::F64 => {
                modulus_in_place::<f64>(&mut tensor.buffer, self.modulus)
            },
            ElementType::Utf8
            | ElementType::Complex128
            | ElementType::Complex64 => {
                return Err(KernelError::InvalidInput(InvalidInput {
                    name: tensor.name,
                    reason: InvalidInputReason::UnsupportedShape,
                }))
            },
        }

        Ok(vec![tensor])
    }
}

fn modulus_in_place<T>(bytes: &mut [u8], modulus: f64)
where
    T: ToPrimitive + FromPrimitive + Copy + ValueType,
{
    let items: &mut [T] = bytes.elements_mut();

    for item in items {
        let result = item
            .to_f64()
            .map(|n| n % modulus)
            .and_then(|n| T::from_f64(n));

        if let Some(updated) = result {
            *item = updated;
        }
    }
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::SliceExt;

    use super::*;

    #[test]
    fn apply_modulus() {
        let mut values = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];

        modulus_in_place::<f64>(values.as_bytes_mut(), 2.0);

        assert_eq!(values, [0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
