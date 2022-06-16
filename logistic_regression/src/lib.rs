use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{Array1, Array2, ArrayView1, ArrayView2},
};
use smartcore::{
    linalg::naive::dense_matrix::*, linear::logistic_regression::*,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Logistic,
}

fn metadata() -> Metadata {
    Metadata::new("Logistic Regression", env!("CARGO_PKG_VERSION"))
        .with_description(
            "a statistical model that models the probability of one event taking place by having the log-odds for the event be a linear combination of one or more independent variables.",
        )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("classification")
        .with_tag("linear modeling")
        .with_tag("analytics")
        .with_input(TensorMetadata::new("x_train"))
        .with_input(TensorMetadata::new("y_train"))
        .with_input(TensorMetadata::new("x_test"))
        .with_output(TensorMetadata::new("y_test"))
}

struct Logistic;

impl ProcBlock for Logistic {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new(
                    "x_train",
                    ElementTypeConstraint::F64,
                    vec![0, 0],
                ),
                TensorConstraint::new(
                    "y_train",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
                TensorConstraint::new(
                    "x_test",
                    ElementTypeConstraint::F64,
                    vec![0, 0],
                ),
            ],
            outputs: vec![TensorConstraint::new(
                "y_test",
                ElementTypeConstraint::F64,
                vec![0],
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let x_train = Tensor::get_named(&inputs, "x_train")?.view_2d()?;
        let y_train = Tensor::get_named(&inputs, "y_train")?.view_1d()?;
        let x_test = Tensor::get_named(&inputs, "x_test")?.view_2d()?;

        let output = transform(x_train, y_train, x_test)?;

        Ok(vec![Tensor::new("y_test", &output)])
    }
}

impl From<Vec<Argument>> for Logistic {
    fn from(_: Vec<Argument>) -> Self { Logistic }
}

fn transform(
    x_train: ArrayView2<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    x_test: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, RunError> {

    let (rows, columns) = x_train.dim();

    let x_train: Vec<f64> = x_train.iter().map(|e| *e as f64).collect();

    let x_train =
        DenseMatrix::from_array(rows, columns, &x_train);

    let y_train: Vec<_> = y_train.to_vec();

    let model = LogisticRegression::fit(&x_train, &y_train, Default::default())
        .map_err(RunError::other)?;

    let (rows, columns) = x_test.dim();

    let x_test: Vec<f64> = x_test.iter().map(|e| *e as f64).collect();

    let x_test =
        DenseMatrix::from_array(rows, columns, &x_test);

    model
        .predict(&x_test)
        .map(Array1::from_vec)
        .map_err(RunError::other)
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray::array;

    use super::*;

    #[test]
    fn check_model() {
        let x_train: Array2<f64> = array![
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1.0],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
        ];
        let y_train: Array1<f64> = array![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
        ];

        let y_pred =
            transform(x_train.view(), y_train.view(), x_train.view()).unwrap();

        assert_eq!(y_pred, y_train);
    }
}
