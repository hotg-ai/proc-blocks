use hotg_rune_proc_blocks::{
    guest::{
        Argument, ArgumentHint, ArgumentMetadata, Dimensions,
        ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::ArrayView1,
};
use smartcore::metrics::*;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Accuracy,
}

fn metadata() -> Metadata {
    Metadata::new("Accuracy", env!("CARGO_PKG_VERSION"))
        .with_description("calculates accuracy of predicted labels when compared to true labels")
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("metric")
        .with_tag("analytics")
        .with_argument(ArgumentMetadata::new("element_type")
        .with_description("The type of tensor this proc-block will accept")
        .with_default_value("f64")
        .with_hint(ArgumentHint::one_of([
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        ]))
    )
    .with_input(TensorMetadata::new("y_true"))
    .with_input(TensorMetadata::new("y_pred"))
    .with_output(TensorMetadata::new("accuracy"))
}

/// A proc block which can perform linear regression
struct Accuracy;

impl ProcBlock for Accuracy {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::numeric("y_true", vec![0]),
                TensorConstraint::numeric("y_pred", vec![0]),
            ],
            outputs: vec![TensorConstraint {
                name: "accuracy".to_string(),
                dimensions: Dimensions::Fixed(vec![1]),
                element_type: ElementTypeConstraint::F64,
            }],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let y_true = Tensor::get_named(&inputs, "y_true")?.view_1d()?;
        let y_pred = Tensor::get_named(&inputs, "y_pred")?.view_1d()?;

        let accuracy = transform(y_true, y_pred);

        Ok(vec![Tensor::new_1d("accuracy", &[accuracy])])
    }
}

impl From<Vec<Argument>> for Accuracy {
    fn from(_: Vec<Argument>) -> Self { Accuracy }
}

fn transform(y_true: ArrayView1<'_, f64>, y_pred: ArrayView1<'_, f64>) -> f64 {
    // Note: We need to unnecessarily copy our inputs here because
    // smartcore's accuracy metric accepts types implementing BaseVector.
    // However, they only have an implementation for Vec<T> and not &[T]
    // or ndarray's 1D arrays.
    let y_true: Vec<f64> = y_true.iter().copied().collect();
    let y_pred: Vec<f64> = y_pred.iter().copied().collect();

    ClassificationMetrics::accuracy().get_score(&y_true, &y_pred)
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn check_transform() {
        let y_pred = ndarray::array![0., 2., 1., 3.];
        let y_true = ndarray::array![0., 1., 2., 3.];

        let accuracy = transform(y_true.view(), y_pred.view());

        assert_eq!(0.5, accuracy);
    }
}
