use hotg_rune_proc_blocks::guest::{
    Argument, ArgumentMetadata, ArgumentType, CreateError, Dimensions,
    ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
    TensorConstraint, TensorConstraints, TensorMetadata,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: BinaryClassification,
}

fn metadata() -> Metadata {
    Metadata::new("Binary Classification", env!("CARGO_PKG_VERSION"))
        .with_description(
            "Classify each element in a tensor depending on whether they are above or below a certain threshold.",
        )
       .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("classify")
        .with_argument(ArgumentMetadata::new("threshold")
        .with_default_value("0.5")
        .with_description("The classification threshold")
    .with_hint(ArgumentType::Float))
    .with_input(TensorMetadata::new("input").with_description("The numbers to classify"))
    .with_output(TensorMetadata::new("classified")
    .with_description("A tensor of `1`'s and `0`'s, where `1` indicates an element was above the `threshold` and `0` means it was below."))
}

/// A proc-block which takes a rank 1 `tensor` as input, return 1 if value
/// inside the tensor is greater than 1 otherwise 0.
struct BinaryClassification {
    threshold: f32,
}

impl ProcBlock for BinaryClassification {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "input",
                ElementTypeConstraint::U32,
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::new(
                "output",
                ElementTypeConstraint::U32,
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "input")?.view::<f32>()?;

        let output =
            tensor.mapv(|v| if v >= self.threshold { 1_u32 } else { 0 });

        Ok(vec![Tensor::new("output", &output)])
    }
}

impl TryFrom<Vec<Argument>> for BinaryClassification {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let threshold = hotg_rune_proc_blocks::guest::parse::optional_arg(
            &args,
            "threshold",
        )?
        .unwrap_or(0.5);

        Ok(BinaryClassification { threshold })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_classification() {
        let transform = BinaryClassification { threshold: 0.5 };
        let inputs = vec![Tensor::new_1d("input", &[0.7_f32])];
        let should_be = vec![Tensor::new_1d("output", &[1_u32])];

        let got = transform.run(inputs).unwrap();

        assert_eq!(got, should_be);
    }
}
