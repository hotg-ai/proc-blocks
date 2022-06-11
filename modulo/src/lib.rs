#![allow(dead_code)]

use hotg_rune_proc_blocks::ndarray::ArrayViewMutD;
use num_traits::{FromPrimitive, ToPrimitive};
use wit_bindgen_rust::Handle;

use crate::proc_block_v2::*;

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v2.wit");

hotg_rune_proc_blocks::generate_support!(crate::proc_block_v2);

pub struct ProcBlockV2;

impl proc_block_v2::ProcBlockV2 for ProcBlockV2 {
    fn metadata() -> Metadata {
        Metadata::new("Modulo", env!("CARGO_PKG_VERSION"))
            .with_description(env!("CARGO_PKG_DESCRIPTION"))
            .with_repository(env!("CARGO_PKG_REPOSITORY"))
            .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
            .with_argument(
                ArgumentMetadata::new("modulus")
                    .with_description("The modulus operand")
                    .with_hint(ArgumentHint::ArgumentType(ArgumentType::Float)),
            )
            .with_input(TensorMetadata::new("input"))
            .with_output(TensorMetadata::new("output"))
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
            inputs: vec![TensorConstraint::numeric(
                "input",
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::numeric(
                "output",
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError> {
        let mut tensor = inputs
            .into_iter()
            .find(|tensor| tensor.name == "input")
            .ok_or_else(|| InvalidInput::not_found("input"))?;

        if let Ok(tensor) = tensor.view_mut::<u8>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<i8>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<u16>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<i16>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<u32>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<i32>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<f32>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<u64>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<i64>() {
            modulo_in_place(tensor, self.modulus);
        } else if let Ok(tensor) = tensor.view_mut::<f64>() {
            modulo_in_place(tensor, self.modulus);
        } else {
            return Err(KernelError::InvalidInput(InvalidInput {
                name: tensor.name,
                reason: InvalidInputReason::UnsupportedShape,
            }));
        }

        Ok(vec![tensor.with_name("output")])
    }
}

fn modulo_in_place<T>(mut array: ArrayViewMutD<'_, T>, modulus: f64)
where
    T: ToPrimitive + FromPrimitive + Copy,
{
    for item in array.iter_mut() {
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
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    fn args(arguments: &[(&str, &str)]) -> Vec<Argument> {
        arguments
            .iter()
            .map(|(n, v)| Argument {
                name: n.to_string(),
                value: v.to_string(),
            })
            .collect()
    }

    #[test]
    fn create_modulo_with_good_modulus() {
        let args = args(&[("modulus", "42.0")]);

        let proc_block = Node::try_from(args).unwrap();

        assert_eq!(proc_block, Node { modulus: 42.0 });
    }

    #[test]
    fn apply_modulus() {
        let mut values = ndarray::array![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];

        modulo_in_place(values.view_mut().into_dyn(), 2.0);

        assert_eq!(values, ndarray::array![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
