use hotg_rune_proc_blocks::{
    guest::{
        Argument, Dimensions, ElementTypeConstraint, InvalidInput, Metadata,
        ProcBlock, RunError, Tensor, TensorConstraint, TensorConstraints,
        TensorMetadata,
    },
    ndarray::ArrayViewMutD,
};
use num_traits::Float;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Softmax,
}

fn metadata() -> Metadata {
    Metadata::new("Softmax", env!("CARGO_PKG_VERSION"))
        .with_description(env!("CARGO_PKG_DESCRIPTION"))
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("softmax")
        .with_tag("image")
        .with_tag("nlp")
        .with_tag("numeric")
        .with_tag("classification")
        .with_input(TensorMetadata::new("input"))
        .with_input(TensorMetadata::new("soft_max").with_description(
            "Vector normalised into probability distribution",
        ))
}

struct Softmax;

impl ProcBlock for Softmax {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "input",
                ElementTypeConstraint::F32 | ElementTypeConstraint::F64,
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::new(
                "soft_max",
                ElementTypeConstraint::F32 | ElementTypeConstraint::F64,
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, mut inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let mut input = Tensor::take_named(&mut inputs, "input")?;

        if let Ok(floats) = input.view_mut::<f32>() {
            softmax_inplace(floats);
        } else if let Ok(doubles) = input.view_mut::<f64>() {
            softmax_inplace(doubles);
        } else {
            return Err(
                InvalidInput::incompatible_element_type(&input.name).into()
            );
        }

        Ok(vec![input.with_name("soft_max")])
    }
}

impl From<Vec<Argument>> for Softmax {
    fn from(_: Vec<Argument>) -> Self { Softmax }
}

fn softmax_inplace<T>(mut input: ArrayViewMutD<'_, T>)
where
    T: Float + num_traits::FromPrimitive,
{
    input.mapv_inplace(|x| x.exp());

    let sum = input.sum();
    if !sum.is_zero() {
        input.mapv_inplace(|x| x / sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

    #[test]
    fn softmax_uniform() {
        let mut input = ndarray::arr1(&[1.0, 1.0, 1.0, 1.0]);
        let softmax_correct = ndarray::arr1(&[0.25, 0.25, 0.25, 0.25]);

        softmax_inplace(input.view_mut().into_dyn());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn softmax_single() {
        let mut input = ndarray::arr1(&[1.0, 0.0]);
        let softmax_correct =
            ndarray::arr1(&[0.7310585786300049, 0.26894142136999510]);
        softmax_inplace(input.view_mut().into_dyn());

        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn known_values() {
        let mut input = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let softmax_correct = ndarray::arr1(&[
            0.09003057317038046,
            0.24472847105479767,
            0.6652409557748219,
        ]);

        softmax_inplace(input.view_mut().into_dyn());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn softmax_zeros() {
        let mut input = ndarray::arr1(&[0.0, 0.0]);
        let softmax_correct = ndarray::arr1(&[0.5, 0.5]);

        softmax_inplace(input.view_mut().into_dyn());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn softmax_zero() {
        let mut input = ndarray::arr1(&[0.0]);
        let softmax_correct = ndarray::arr1(&[1.0]);

        softmax_inplace(input.view_mut().into_dyn());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn softmax_empty() {
        let empty: &[f32] = &[];
        let mut input = ndarray::Array::from_vec(empty.to_vec());
        let softmax_correct = ndarray::Array::from_vec(empty.to_vec());

        softmax_inplace(input.view_mut().into_dyn());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn floats() {
        let inputs = vec![Tensor::new_1d("input", &[1.0_f32, 2.0, 3.0])];
        let softmax_correct = ndarray::arr1(&[
            0.09003057317038046_f32,
            0.24472847105479767,
            0.6652409557748219,
        ]);

        let got = Softmax.run(inputs).unwrap();

        assert_eq!(got, vec![Tensor::new("soft_max", &softmax_correct)]);
    }
}
