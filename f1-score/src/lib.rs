use hotg_rune_proc_blocks::guest::{
    Argument, ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
    TensorConstraint, TensorConstraints, TensorMetadata,
};
use smartcore::metrics::{f1::F1, precision::Precision, recall::Recall};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: F1Score,
}

fn metadata() -> Metadata {
    Metadata::new("F-Score", env!("CARGO_PKG_VERSION"))
        .with_description("for assessing prediction error")
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("metric")
        .with_tag("analytics")
        .with_input(TensorMetadata::new("y_true"))
        .with_input(TensorMetadata::new("y_pred"))
        .with_output(TensorMetadata::new("f1_score"))
        .with_output(TensorMetadata::new("precision"))
        .with_output(TensorMetadata::new("recall"))
}

/// A proc-block used to calculate f1-score
struct F1Score;

impl ProcBlock for F1Score {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new(
                    "y_true",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
                TensorConstraint::new(
                    "y_pred",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
            ],
            outputs: vec![
                TensorConstraint::new(
                    "f1_score",
                    ElementTypeConstraint::F64,
                    vec![1],
                ),
                TensorConstraint::new(
                    "precision",
                    ElementTypeConstraint::F64,
                    vec![1],
                ),
                TensorConstraint::new(
                    "recall",
                    ElementTypeConstraint::F64,
                    vec![1],
                ),
            ],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let y_true = Tensor::get_named(&inputs, "y_true")?.view_1d()?;
        let y_pred = Tensor::get_named(&inputs, "y_pred")?.view_1d()?;

        let (f1_score, precision, recall) =
            transform(y_true.to_vec(), y_pred.to_vec())?;

        Ok(vec![
            Tensor::new_1d("f1_score", &[f1_score]),
            Tensor::new_1d("precision", &[precision]),
            Tensor::new_1d("recall", &[recall]),
        ])
    }
}

impl From<Vec<Argument>> for F1Score {
    fn from(_: Vec<Argument>) -> Self { F1Score }
}

fn transform(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
) -> Result<(f64, f64, f64), RunError> {
    if y_true.len() != y_pred.len() {
        let msg = format!(
            "\"y_true\" and \"y_pred\" should have the same dimensions ({} != {})",
            y_true.len(), y_pred.len(),
    );
        return Err(RunError::other(msg));
    }

    let f1 = F1 { beta: 1.0 }.get_score(&y_pred, &y_true);
    let precision = Precision {}.get_score(&y_pred, &y_true);
    let recall = Recall {}.get_score(&y_pred, &y_true);

    Ok((f1, precision, recall))
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn known_inputs() {
        let y_pred = ndarray::array![0_f64, 0., 1., 1., 1., 1.];
        let y_true = ndarray::array![0_f64, 1., 1., 0., 1., 0.];
        let inputs = vec![
            Tensor::new("y_pred", &y_pred),
            Tensor::new("y_true", &y_true),
        ];

        let got = F1Score.run(inputs).unwrap();

        let should_be = vec![
            Tensor::new_1d("f1_score", &[0.5714285714285715_f64]),
            Tensor::new_1d("precision", &[0.6666666666666666_f64]),
            Tensor::new_1d("recall", &[0.5_f64]),
        ];
        assert_eq!(got, should_be);
    }

    #[test]
    fn check_f1() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let (f1, precision, recall) = transform(y_true, y_pred).unwrap();

        assert_eq!(0.5714285714285715, f1);
        assert_eq!(0.6666666666666666, precision);
        assert_eq!(0.5, recall);
    }
}
