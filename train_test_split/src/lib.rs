use hotg_rune_proc_blocks::{
    guest::{
        Argument, ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata, ArgumentMetadata, parse, ArgumentType
    },
    ndarray::{Array1, ArrayView1, ArrayView2},
};
use smartcore::{linalg::naive::dense_matrix::*, linear::elastic_net::*};

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
    test_size: f32
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
                )
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
            ]
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {

        let features = Tensor::get_named(&inputs, "features")?.view_2d()?;
        let targets = Tensor::get_named(&inputs, "targets")?.view_1d()?;

        let count = parse::required_arg(&vec![Argument{ name: "test_size".to_string(), value: self.count.to_string() }], "test_size").unwrap();

        let (x_train, y_train, x_test, y_test) = transform(features, y, test_size);

        Ok(vec![Tensor::new("x_train", &x_train), Tensor::new("y_train", &y_train), Tensor::new("x_test", &x_test), Tensor::new("y_test", &y_test)])

    }
}


fn transform(
    x: &[f64],
    y: Vec<f64>,
    test_size: f32,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>
) {
    let x = DenseMatrix::from_array(x_dim[0] as usize, x_dim[1] as usize, x);

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, test_size, false);
    let x_train: Vec<f64> = x_train.iter().map(|f| f).collect();
    let x_test: Vec<f64> = x_test.iter().map(|f| f).collect();

    (x_train, y_train, x_test, y_test)
}

impl TryFrom<Vec<Argument>> for TrainTestSplit {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let test_size = hotg_rune_proc_blocks::guest::parse::optional_arg(
            &args,
            "test_size",
        )?
        .unwrap_or(0.2);

        Ok(TrainTestSplit { test_size })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn check_test_dim() {
//         let x = [
//             5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4, 5.1,
//             3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4,
//         ];
//         let y: Vec<f64> = vec![0., 0., 1., 0., 0., 1.];

//         let dim: Vec<u32> = vec![6, 4];

//         let (_x_train, x_test, _y_train, _y_test) =
//             transform(&x,  y, 0.2);

//         let test_dim = x_test.dim();

//         let should_be = (1, 4);

//         assert_eq!(test_dim, should_be);
//     }
//     #[test]
//     fn check_train_dim() {
//         let x = [
//             5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4, 5.1,
//             3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4,
//         ];
//         let y: Vec<f64> = vec![0., 0., 1., 0., 0., 1.];

//         let dim: Vec<u32> = vec![6, 4];

//         let (x_train, _x_test, _y_train, _y_test, train_dim, _test_dim) =
//             transform(&x, y, 0.2);

//         let should_be = (5, 4);

//         let train_dim = x_train.dim();

//         assert_eq!(train_dim, should_be);
//     }
// }
