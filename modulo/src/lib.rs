#![allow(dead_code)]

use hotg_rune_proc_blocks::{
    guest::{
        Argument, ArgumentHint, ArgumentMetadata, ArgumentType, CreateError,
        Dimensions, InvalidInput, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::ArrayViewMutD,
};
use num_traits::{FromPrimitive, ToPrimitive};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Modulo,
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct Modulo {
    modulus: f64,
}

impl ProcBlock for Modulo {
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

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
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
            return Err(
                InvalidInput::incompatible_element_type(tensor.name).into()
            );
        }

        Ok(vec![tensor.with_name("output")])
    }
}

impl TryFrom<Vec<Argument>> for Modulo {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let modulus = hotg_rune_proc_blocks::guest::parse::required_arg(
            &args, "modulus",
        )?;

        Ok(Modulo { modulus })
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
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

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

        let proc_block = Modulo::try_from(args).unwrap();

        assert_eq!(proc_block, Modulo { modulus: 42.0 });
    }

    #[test]
    fn apply_modulus() {
        let inputs = vec![Tensor::new(
            "input",
            &ndarray::array![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        )];
        let expected = vec![Tensor::new(
            "output",
            &ndarray::array![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0],
        )];
        let modulo = Modulo { modulus: 2.0 };

        let outputs = modulo.run(inputs).unwrap();

        assert_eq!(outputs, expected);
    }
}
