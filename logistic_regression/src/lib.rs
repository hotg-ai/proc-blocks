use hotg_rune_proc_blocks::{
    guest::{
        parse, Argument, ArgumentMetadata, ArgumentType, CreateError, Dimensions,
        ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor, TensorConstraint,
        TensorConstraints, TensorMetadata,
    },
    ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2},
};
use smartcore::{
    linalg::naive::dense_matrix::*,
    linear::logistic_regression::*,
    metrics::{f1::F1, precision::Precision, recall::Recall, *},
    model_selection::train_test_split,
};

use serde_json;

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
        .with_argument(ArgumentMetadata::new("test_size")
        .with_default_value("0.2")
        .with_description("the proportion of the dataset to include in the test split")
        .with_hint(ArgumentType::Float))
        .with_input(TensorMetadata::new("features").with_description("features"))
        .with_input(TensorMetadata::new("targets").with_description("targets"))
        .with_output(TensorMetadata::new("model"))
        .with_output(TensorMetadata::new("accuracy"))
        .with_output(TensorMetadata::new("f1"))
        .with_output(TensorMetadata::new("precision"))
        .with_output(TensorMetadata::new("recall"))
}

// use serde::{Deserialize, Serialize};
// use serde_json;

struct Logistic {
    test_size: f32,
}

impl ProcBlock for Logistic {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new("features", ElementTypeConstraint::F64, vec![0, 0]),
                TensorConstraint::new("targets", ElementTypeConstraint::F64, vec![0]),
            ],
            outputs: vec![
                TensorConstraint::new("model", ElementTypeConstraint::UTF8, Dimensions::Dynamic),
                TensorConstraint::new("accuracy", ElementTypeConstraint::F64, vec![1]),
                TensorConstraint::new("f1", ElementTypeConstraint::F64, vec![1]),
                TensorConstraint::new("precision", ElementTypeConstraint::F64, vec![1]),
                TensorConstraint::new("recall", ElementTypeConstraint::F64, vec![1]),
            ],
        }
    }

    fn run (&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let features = Tensor::get_named(&inputs, "features")?.view_2d()?;
        let targets = Tensor::get_named(&inputs, "targets")?.view_1d()?;

        let (model, accuracy, f1, precision, recall) = transform(features, targets, self.test_size)?;

        Ok(vec![
            Tensor::from_strings("model", &model),
            Tensor::new_1d("accuracy", &[accuracy]),
            Tensor::new_1d("f1", &[f1]),
            Tensor::new_1d("precision", &[precision]),
            Tensor::new_1d("recall", &[recall]),
        ])
    }
}


fn transform(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    test_size: f32,
) -> Result<(String, f64, f64, f64, f64), RunError> {
    let (rows, columns) = x.dim();
    let x = DenseMatrix::new(rows, columns, x.into_iter().copied().collect());

    let y = y.to_vec();

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, test_size, false);

    let x_train: Array2<f64> =
        Array::from_shape_vec(x_train.shape(), x_train.iter().collect()).unwrap();
    let x_test: Array2<f64> =
        Array::from_shape_vec(x_test.shape(), x_test.iter().collect()).unwrap();
    let y_train: Array1<f64> = Array::from_shape_vec(y_train.len(), y_train).unwrap();
    let y_test: Array1<f64> = Array::from_shape_vec(y_test.len(), y_test).unwrap();

    let (rows, columns) = x_train.dim();
    let x_train: Vec<f64> = x_train.t().iter().copied().collect();
    let x_train = DenseMatrix::new(rows, columns, x_train);

    let y_train: Vec<_> = y_train.to_vec();

    let model =
        LogisticRegression::fit(&x_train, &y_train, Default::default()).map_err(RunError::other)?;

    let a = model.coefficients();

    let (rows, columns) = x_test.dim();
    let x_test: Vec<f64> = x_test.t().iter().copied().collect();
    let x_test = DenseMatrix::new(rows, columns, x_test);

    let y_pred = model
        .predict(&x_test)
        .map(Array1::from_vec)
        .map_err(RunError::other)?;

    if y_test.len() != y_pred.len() {
        let msg = format!(
            "\"y_true\" and \"y_pred\" should have the same dimensions ({} != {})",
            y_test.len(),
            y_pred.len(),
        );
        return Err(RunError::other(msg));
    }

    let model = serde_json::to_string(&model).map_err(RunError::other);
    let accuracy = ClassificationMetrics::accuracy().get_score(&y_test.to_vec(), &y_pred.to_vec());
    let f1 = F1 { beta: 1.0 }.get_score(&y_test.to_vec(), &y_pred.to_vec());
    let precision = Precision {}.get_score(&y_test.to_vec(), &y_pred.to_vec());
    let recall = Recall {}.get_score(&y_test.to_vec(), &y_pred.to_vec());

    Ok((model, accuracy, f1, precision, recall))
}

impl TryFrom<Vec<Argument>> for Logistic {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let test_size = parse::optional_arg(&args, "test_size")?.unwrap_or(0.2);

        Ok(Logistic { test_size })
    }
}

// #[cfg(test)]
// mod tests {
//     use hotg_rune_proc_blocks::ndarray::array;

//     use super::*;

//     #[test]
//     fn check_model() {
//         let x_train: Array2<f64> = array![
//             [5.1, 3.5, 1.4, 0.2],
//             [4.9, 3.0, 1.4, 0.2],
//             [4.7, 3.2, 1.3, 0.2],
//             [4.6, 3.1, 1.5, 0.2],
//             [5.0, 3.6, 1.4, 0.2],
//             [5.4, 3.9, 1.7, 0.4],
//             [4.6, 3.4, 1.4, 0.3],
//             [5.0, 3.4, 1.5, 0.2],
//             [4.4, 2.9, 1.4, 0.2],
//             [4.9, 3.1, 1.5, 0.1],
//             [7.0, 3.2, 4.7, 1.4],
//             [6.4, 3.2, 4.5, 1.5],
//             [6.9, 3.1, 4.9, 1.5],
//             [5.5, 2.3, 4.0, 1.3],
//             [6.5, 2.8, 4.6, 1.5],
//             [5.7, 2.8, 4.5, 1.3],
//             [6.3, 3.3, 4.7, 1.6],
//             [4.9, 2.4, 3.3, 1.0],
//             [6.6, 2.9, 4.6, 1.3],
//             [5.2, 2.7, 3.9, 1.4],
//         ];
//         let y_train: Array1<f64> = array![
//             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
//             1., 1., 1.,
//         ];

//         let y_pred =
//             transform(x_train.view(), y_train.view(), x_train.view()).unwrap();

//         assert_eq!(y_pred, y_train);
//     }
// }
