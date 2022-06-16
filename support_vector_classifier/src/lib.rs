use hotg_rune_proc_blocks::{
    guest::{
        parse, Argument, ArgumentMetadata, ArgumentType, CreateError,
        ElementTypeConstraint, Metadata, ProcBlock, RunError, Tensor,
        TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{Array1, ArrayView1, ArrayView2},
};
use smartcore::{
    linalg::naive::dense_matrix::*,
    svm::{
        svc::{SVCParameters, SVC},
        Kernels,
    },
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: SupportVectorClassifier,
}

fn metadata() -> Metadata {
    // TODO: how to add an array of string: [linear, rbf, polynomial,
    // polynomial_with_degree, sigmoid, sigmoiod_with_gamma].
    // Have to figure out how to how to change the parameter of polynomial,
    // sigmoid, etc

    Metadata::new(" Support Vector Classifier", env!("CARGO_PKG_VERSION"))
    .with_description(
            "a binary approach for modelling the relationship between a scalar response and one or more explanatory variables",
        )
    .with_repository(env!("CARGO_PKG_REPOSITORY"))
    .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
    .with_tag("binary classifier")
    .with_tag("analytics")
    .with_argument(
        ArgumentMetadata::new("epochs")
            .with_description("Number of epochs")
            .with_hint(ArgumentType::Integer)
            .with_default_value("5"),
    )
    .with_argument(
        ArgumentMetadata::new("c")
            .with_description("Penalizing parameter")
            .with_hint(ArgumentType::Float)
            .with_default_value("200.0"),
    )
    .with_argument(
        ArgumentMetadata::new("tol")
            .with_description("Tolerance for stopping criterion")
            .with_hint(ArgumentType::Float)
            .with_default_value("0.001"),
    )
    .with_input(TensorMetadata::new("x_train"))
    .with_input(TensorMetadata::new("y_train"))
    .with_input(TensorMetadata::new("x_test"))
    .with_output(TensorMetadata::new("y_test"))
}

/// a binary classifier that uses an optimal hyperplane to separate the points
/// in the input variable space by their class.
struct SupportVectorClassifier {
    epochs: u32,
    c: f64,
    tol: f64,
}

impl ProcBlock for SupportVectorClassifier {
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

        let output =
            transform(x_train, y_train, x_test, self.c, self.epochs, self.tol)?;

        Ok(vec![Tensor::new("y_train", &output)])
    }
}

impl TryFrom<Vec<Argument>> for SupportVectorClassifier {
    type Error = CreateError;

    fn try_from(value: Vec<Argument>) -> Result<Self, Self::Error> {
        let epochs = parse::optional_arg(&value, "epochs")?.unwrap_or(5);
        let c = parse::optional_arg(&value, "c")?.unwrap_or(200.0);
        let tol = parse::optional_arg(&value, "tol")?.unwrap_or(0.001);

        Ok(SupportVectorClassifier { epochs, c, tol })
    }
}

fn transform(
    x_train: ArrayView2<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    x_test: ArrayView2<'_, f64>,
    c: f64,
    epoch: u32,
    tol: f64,
) -> Result<Array1<f64>, RunError> {
    // todo: let user change the kernel. Right now setting it to 'linear'
    let svc_parameters = SVCParameters::default()
        .with_c(c)
        .with_epoch(epoch.try_into().unwrap())
        .with_kernel(Kernels::linear())
        .with_tol(tol);

    let (rows, columns) = x_train.dim();
    let x_train =
        DenseMatrix::new(rows, columns, x_train.iter().copied().collect());

    let model = SVC::fit(&x_train, &y_train.to_vec(), svc_parameters)
        .map_err(RunError::other)?;

    let (rows, columns) = x_test.dim();
    let x_test =
        DenseMatrix::new(rows, columns, x_test.iter().copied().collect());

    model
        .predict(&x_test)
        .map(Array1::from_vec)
        .map_err(RunError::other)
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn check_model() {
        let x_train = ndarray::array![
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
        let y_train = ndarray::array![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
        ];
        let svc = SupportVectorClassifier {
            epochs: 5,
            c: 200.0,
            tol: 0.001,
        };
        let inputs = vec![
            Tensor::new("x_train", &x_train),
            Tensor::new("y_train", &y_train),
            Tensor::new("x_test", &x_train),
        ];

        let got = svc.run(inputs).unwrap();

        let should_be = vec![Tensor::new("y_train", &y_train)];
        assert_eq!(got, should_be);
    }
}
