use hotg_rune_proc_blocks::guest::{
    Argument, ElementType, Metadata, ProcBlock, RunError, Tensor,
    TensorConstraint, TensorConstraints, TensorMetadata,
};
use smartcore::metrics::{
    mean_absolute_error::MeanAbsoluteError, mean_squared_error::MeanSquareError,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: PredictionErrors,
}

fn metadata() -> Metadata {
    Metadata::new("Errors", env!("CARGO_PKG_VERSION"))
        .with_description("for assessing prediction error")
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("metric")
        .with_tag("analytics")
        .with_tag("loss")
        .with_input(TensorMetadata::new("y_true"))
        .with_input(TensorMetadata::new("y_pred"))
        .with_output(TensorMetadata::new("mean_absolute_error"))
        .with_output(TensorMetadata::new("mean_square_error"))
}

/// a proc-block to find Mean Absolute Error and Mean Squared Error
#[derive(Debug, Default, Clone, PartialEq)]
struct PredictionErrors;

impl ProcBlock for PredictionErrors {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new("y_true", ElementType::F64, [0]),
                TensorConstraint::new("y_pred", ElementType::F64, [0]),
            ],
            outputs: vec![
                TensorConstraint::new(
                    "mean_absolute_error",
                    ElementType::F64,
                    [1],
                ),
                TensorConstraint::new(
                    "mean_square_error",
                    ElementType::F64,
                    [1],
                ),
            ],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let y_true = Tensor::get_named(&inputs, "y_true")?.view_1d::<f64>()?;
        let y_pred = Tensor::get_named(&inputs, "y_pred")?.view_1d::<f64>()?;

        let y_pred = y_pred.to_vec();
        let y_true = y_true.to_vec();
        let mae = MeanAbsoluteError {}.get_score(&y_true, &y_pred);
        let mse = MeanSquareError {}.get_score(&y_true, &y_pred);

        Ok(vec![
            Tensor::new_1d("mean_absolute_error", &[mae]),
            Tensor::new_1d("mean_square_error", &[mse]),
        ])
    }
}

impl From<Vec<Argument>> for PredictionErrors {
    fn from(_: Vec<Argument>) -> Self { PredictionErrors::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let predict_errors = PredictionErrors;
        let inputs = vec![
            Tensor::new_1d("y_pred", &[0.0_f64, 0.0, 1.0, 1.0, 1.0, 1.0]),
            Tensor::new_1d("y_true", &[0.0_f64, 1.0, 1.0, 0.0, 1.0, 0.0]),
        ];

        let got = predict_errors.run(inputs).unwrap();

        let should_be = vec![
            Tensor::new_1d("mean_absolute_error", &[0.5]),
            Tensor::new_1d("mean_square_error", &[0.5]),
        ];
        assert_eq!(got, should_be);
    }
}
