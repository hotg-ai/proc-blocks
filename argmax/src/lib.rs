use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementType, InvalidInput, Metadata, ProcBlock, RunError,
        Tensor, TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::ArrayViewD,
};
use std::{cmp::Ordering, convert::TryFrom};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: ArgMax,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct ArgMax;

impl From<Vec<Argument>> for ArgMax {
    fn from(_: Vec<Argument>) -> Self { ArgMax }
}

fn metadata() -> Metadata {
    Metadata::new("Arg Max", env!("CARGO_PKG_VERSION"))
        .with_description(env!("CARGO_PKG_DESCRIPTION"))
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("max")
        .with_tag("index")
        .with_tag("numeric")
        .with_input(TensorMetadata::new("input"))
        .with_output(TensorMetadata::new("max_index").with_description(
            "The index of the element with the highest value",
        ))
}

impl ProcBlock for ArgMax {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::numeric("input", vec![0])],
            outputs: vec![TensorConstraint::numeric("max_index", vec![1])],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "input")?;

        let index = match tensor.element_type {
            ElementType::U8 => arg_max(tensor.view::<u8>()?),
            ElementType::I8 => arg_max(tensor.view::<i8>()?),
            ElementType::U16 => arg_max(tensor.view::<u16>()?),
            ElementType::I16 => arg_max(tensor.view::<i16>()?),
            ElementType::U32 => arg_max(tensor.view::<u32>()?),
            ElementType::I32 => arg_max(tensor.view::<i32>()?),
            ElementType::F32 => arg_max(tensor.view::<f32>()?),
            ElementType::U64 => arg_max(tensor.view::<u64>()?),
            ElementType::I64 => arg_max(tensor.view::<i64>()?),
            ElementType::F64 => arg_max(tensor.view::<f64>()?),
            _ => {
                return Err(InvalidInput::incompatible_element_type(
                    &tensor.name,
                )
                .into());
            },
        };

        let index = index
            .ok_or_else(|| RunError::other("The input tensor was empty"))?;

        Ok(vec![Tensor::new_1d("max_index", &[index as u32])])
    }
}

fn arg_max<T>(values: ArrayViewD<'_, T>) -> Option<usize>
where
    T: PartialOrd,
{
    let (index, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))?;

    Some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let inputs = vec![Tensor::new_1d("input", &[2.3, 12.4, 55.1, 15.4])];
        let should_be = vec![Tensor::new_1d("max_index", &[2_u32])];

        let got = ArgMax.run(inputs).unwrap();

        assert_eq!(got, should_be);
    }

    #[test]
    fn empty_inputs_are_an_error() {
        let empty: &[f32] = &[];
        let inputs = vec![Tensor::new_1d("input", empty)];

        let error = ArgMax.run(inputs).unwrap_err();

        assert_eq!(error, RunError::other("The input tensor was empty"));
    }
}
