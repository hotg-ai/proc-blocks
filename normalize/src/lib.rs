use hotg_rune_proc_blocks::{
    guest::{
        Argument, Dimensions, InvalidInput, Metadata, ProcBlock, RunError,
        Tensor, TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{ArrayD, ArrayViewD},
};
use num_traits::ToPrimitive;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Normalize,
}

fn metadata() -> Metadata {
    Metadata::new("Normalize", env!("CARGO_PKG_VERSION"))
        .with_description(
            "Normalize a tensor's elements to the range, `[0, 1]`.",
        )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("normalize")
        .with_input(TensorMetadata::new("input"))
        .with_output(
            TensorMetadata::new("normalized")
                .with_description("normalized tensor in the range [0, 1]"),
        )
}

/// Normalize the input to the range `[0, 1]`.
struct Normalize;

impl ProcBlock for Normalize {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::numeric(
                "input",
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::numeric(
                "normalized",
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "input")?;

        let normalized = if let Ok(tensor) = tensor.view::<u8>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<i8>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<u16>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<i16>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<u32>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<i32>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<u64>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<i64>() {
            normalize(tensor)
        } else if let Ok(tensor) = tensor.view::<f64>() {
            normalize(tensor)
        } else {
            return Err(
                InvalidInput::incompatible_element_type(&tensor.name).into()
            );
        };

        Ok(vec![Tensor::new("normalized", &normalized)])
    }
}

impl From<Vec<Argument>> for Normalize {
    fn from(_: Vec<Argument>) -> Self { Normalize }
}

fn normalize<T>(input: ArrayViewD<'_, T>) -> ArrayD<f32>
where
    T: ToPrimitive,
{
    if input.is_empty() {
        return ArrayD::zeros(input.shape());
    }

    let (min, max) =
        input.fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), elem| {
            match elem.to_f32() {
                Some(elem) => (min.min(elem), max.max(elem)),
                None => (min, max),
            }
        });

    let range = max - min;

    if range == 0.0 {
        return ArrayD::zeros(input.shape());
    }

    let mean = (max + min) / 2.0;

    input.map(|v| match v.to_f32() {
        Some(elem) => (elem - min) / range,
        None => mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let inputs = vec![Tensor::new_1d("input", &[0.0_f64, 1.0, 2.0])];

        let output = Normalize.run(inputs).unwrap();

        assert_eq!(
            output,
            vec![Tensor::new_1d("normalized", &[0.0_f32, 0.5, 1.0])]
        );
    }

    #[test]
    fn handle_all_zeroes() {
        let inputs = vec![Tensor::new_1d("input", &[0_i32; 64])];

        let output = Normalize.run(inputs).unwrap();

        assert_eq!(output, vec![Tensor::new_1d("normalized", &[0_f32; 64])]);
    }

    #[test]
    fn empty_input() {
        let inputs = vec![Tensor::new_1d::<i16>("input", &[])];

        let output = Normalize.run(inputs).unwrap();

        assert_eq!(output, vec![Tensor::new_1d::<f32>("normalized", &[])]);
    }
}
