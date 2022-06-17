use hotg_rune_proc_blocks::{
    guest::{
        parse, Argument, ArgumentMetadata, ArgumentType, CreateError,
        ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2},
};
use smartcore::{
    linalg::naive::dense_matrix::*, model_selection::train_test_split,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: TrainTestSplit,
}

fn metadata() -> Metadata {
    Metadata::new("Train Test Split", env!("CARGO_PKG_VERSION"))
        .with_description(
            "a random split into training and test sets",
        )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("split")
        .with_tag("data processing")
        .with_tag("analytics")
        .with_argument(ArgumentMetadata::new("test_size")
        .with_default_value("0.2")
        .with_description("the proportion of the dataset to include in the test split")
        .with_hint(ArgumentType::Float))
        .with_input(TensorMetadata::new("features").with_description("features"))
        .with_input(TensorMetadata::new("targets").with_description("targets"))
        .with_output(TensorMetadata::new("x_train").with_description("training features"))
        .with_output(TensorMetadata::new("y_train").with_description("training labels"))
        .with_output(TensorMetadata::new("x_test").with_description("testing features"))
        .with_output(TensorMetadata::new("y_test").with_description("testing labels"))
}

struct TrainTestSplit {
    test_size: f32,
}

impl ProcBlock for TrainTestSplit {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![
                TensorConstraint::new(
                    "features",
                    ElementTypeConstraint::F64,
                    vec![0, 0],
                ),
                TensorConstraint::new(
                    "targets",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
            ],
            outputs: vec![
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
                TensorConstraint::new(
                    "y_test",
                    ElementTypeConstraint::F64,
                    vec![0],
                ),
            ],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let features = Tensor::get_named(&inputs, "features")?.view_2d()?;
        let targets = Tensor::get_named(&inputs, "targets")?.view_1d()?;

        let (x_train, y_train, x_test, y_test) =
            transform(features, targets, self.test_size);

        Ok(vec![
            Tensor::new("x_train", &x_train),
            Tensor::new("y_train", &y_train),
            Tensor::new("x_test", &x_test),
            Tensor::new("y_test", &y_test),
        ])
    }
}

fn transform(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    test_size: f32,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let (rows, columns) = x.dim();
    let x = DenseMatrix::new(rows, columns, x.into_iter().copied().collect());

    let y = y.to_vec();

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, test_size, false);

    let x_train: Array2<f64> =
        Array::from_shape_vec(x_train.shape(), x_train.iter().collect())
            .unwrap();
    let x_test: Array2<f64> =
        Array::from_shape_vec(x_test.shape(), x_test.iter().collect()).unwrap();
    let y_train: Array1<f64> =
        Array::from_shape_vec(y_train.len(), y_train).unwrap();
    let y_test: Array1<f64> =
        Array::from_shape_vec(y_test.len(), y_test).unwrap();

    (x_train, y_train, x_test, y_test)
}

impl TryFrom<Vec<Argument>> for TrainTestSplit {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let test_size = parse::optional_arg(&args, "test_size")?.unwrap_or(0.2);

        Ok(TrainTestSplit { test_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray::array;

    #[test]
    fn check_test_dim() {
        let x: Array2<f64> = array![
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [5.2, 2.7, 3.9, 1.4],
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [5.2, 2.7, 3.9, 1.4]
        ];
        let y: Array1<f64> = array![0., 0., 1., 0., 0., 1.];

        let (_x_train, _y_train, x_test, _y_test) =
            transform(x.view(), y.view(), 0.2);

        assert_eq!(x_test.dim(), (1, 4));
    }

    #[test]
    fn check_train_dim() {
        let x: Array2<f64> = array![
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [5.2, 2.7, 3.9, 1.4],
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [5.2, 2.7, 3.9, 1.4]
        ];
        let y: Array1<f64> = array![0., 0., 1., 0., 0., 1.];

        let (x_train, y_train, _x_test, _y_test) =
            transform(x.view(), y.view(), 0.2);

        assert_eq!(x_train.dim(), (5, 4));
        assert_eq!(y_train, array![0.0, 1.0, 0.0, 0.0, 1.0]);
    }
}
