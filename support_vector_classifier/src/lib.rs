use hotg_rune_proc_blocks::{ndarray, runtime_v1};
use smartcore::{
    linalg::naive::dense_matrix::*,
    svm::{
        svc::{SVCParameters, SVC},
        Kernels,
    },
};
use std::{convert::TryInto, fmt::Display, str::FromStr};

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// a binary classifier that uses an optimal hyperplane to separate the points
/// in the input variable space by their class.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new(
            " Support Vector Classifier",
            env!("CARGO_PKG_VERSION"),
        );
        metadata.set_description(
            "a binary approach for modelling the relationship between a scalar response and one or more explanatory variables",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("binary classifier");
        metadata.add_tag("analytics");

        let epochs = ArgumentMetadata::new("epochs");
        epochs.set_description("Number of epochs");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Integer);
        epochs.add_hint(&hint);
        epochs.set_default_value("5");
        metadata.add_argument(&epochs);

        let c = ArgumentMetadata::new("c");
        c.set_description("Penalizing parameter");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        c.add_hint(&hint);
        c.set_default_value("200.0");
        metadata.add_argument(&c);

        let tol = ArgumentMetadata::new("tolerance");
        tol.set_description("Tolerance for stopping criterion");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        tol.add_hint(&hint);
        tol.set_default_value("0.001");
        metadata.add_argument(&tol);

        // todo: how to add an array of string: [linear, rbf, polynomial,
        // polynomial_with_degree, sigmoid, sigmoiod_with_gamma].
        // Have to figure out how to how to change the parameter of polynomial,
        // sigmoid, etc

        // let kernel = ArgumentMetadata::new("kernel");
        // epochs.set_description(
        //     "Tolerance for stopping criterion",
        // );
        // let hint = runtime_v1::supported_argument_type(ArgumentType::String);
        // kernel.add_hint(&hint);
        // kernel.set_default_value("linear");
        // metadata.add_argument(&kernel);

        let x_train = TensorMetadata::new("x_train");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_train.add_hint(&hint);
        metadata.add_input(&x_train);

        let y_train = TensorMetadata::new("y_train");
        let hint =
            supported_shapes(&[ElementType::F64], DimensionsParam::Fixed(&[0]));
        y_train.add_hint(&hint);
        metadata.add_input(&y_train);

        let x_test = TensorMetadata::new("x_test");
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_test.add_hint(&hint);
        metadata.add_input(&x_test);

        let y_test = TensorMetadata::new("y_test");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y_test.add_hint(&hint);
        metadata.add_output(&y_test);
        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "x_train",
            ElementType::F64,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_input_tensor(
            "y_train",
            ElementType::F64,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "x_test",
            ElementType::F64,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_output_tensor(
            "y_test",
            ElementType::F64,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let epoch: u32 = get_args("epochs", |n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let c: f64 = get_args("c", |n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let tol: f64 = get_args("tolerance", |n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        // let _kernel: String  = get_args("kernel", |n| ctx.get_argument(n))
        // .map_err(KernelError::InvalidArgument)?;

        let x_train = ctx.get_input_tensor("x_train").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "x_train".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;
        let _xtrain: ndarray::ArrayView2<f64> = x_train
            .buffer
            .view(&x_train.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "x_train".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

        let y_train = ctx.get_input_tensor("y_train").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "y_train".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;
        let _ytrain: ndarray::ArrayView1<f64> = y_train
            .buffer
            .view(&y_train.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "y_train".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

        let x_test = ctx.get_input_tensor("x_test").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "x_test".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;
        let _xtest: ndarray::ArrayView2<f64> = x_test
            .buffer
            .view(&x_test.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "x_test".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

        if x_train.element_type != ElementType::F64
            || y_train.element_type != ElementType::F64
            || x_test.element_type != ElementType::F64
        {
            return Err(KernelError::Other(format!(
                "This proc-block only support f64 element type",
            )));
        }

        let output = transform(
            &x_train.buffer.elements(),
            &x_train.dimensions,
            &y_train.buffer.elements(),
            &x_test.buffer.elements(),
            &x_test.dimensions,
            c,
            epoch,
            tol,
        )?;

        let y_test_dimension = [x_test.dimensions[0]];

        ctx.set_output_tensor(
            "y_test",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &y_test_dimension,
                buffer: &output.to_vec().as_bytes(),
            },
        );

        Ok(())
    }
}

fn get_args<T>(
    name: &str,
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<T, InvalidArgument>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    get_argument(name)
        .ok_or_else(|| InvalidArgument::not_found(name))?
        .parse::<T>()
        .map_err(|e| InvalidArgument::invalid_value(name, e))
}

impl InvalidArgument {
    fn not_found(name: impl Into<String>) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::NotFound,
        }
    }

    fn invalid_value(name: impl Into<String>, reason: impl Display) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::InvalidValue(reason.to_string()),
        }
    }
}

fn transform(
    x_train: &[f64],
    x_train_dim: &[u32],
    y_train: &[f64],
    x_test: &[f64],
    x_test_dim: &[u32],
    c: f64,
    epoch: u32,
    tol: f64,
) -> Result<Vec<f64>, KernelError> {
    // todo: let user change the kernel. Right now setting it to 'linear'
    let svc_parameters = SVCParameters::default()
        .with_c(c)
        .with_epoch(epoch.try_into().unwrap())
        .with_kernel(Kernels::linear())
        .with_tol(tol);

    let x_train = DenseMatrix::from_array(
        x_train_dim[0] as usize,
        x_train_dim[1] as usize,
        x_train,
    );

    let model = SVC::fit(&x_train, &y_train.to_vec(), svc_parameters)
        .map_err(|e| KernelError::Other(e.to_string()))?;

    let x_test = DenseMatrix::from_array(
        x_test_dim[0] as usize,
        x_test_dim[1] as usize,
        x_test,
    );

    model
        .predict(&x_test)
        .map_err(|e| KernelError::Other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_model() {
        let x_train = vec![
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6,
            3.1, 1.5, 0.2, 5.0, 3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4,
            1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5,
            0.1, 7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5,
            5.5, 2.3, 4.0, 1.3, 6.5, 2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3,
            3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7,
            3.9, 1.4,
        ];
        let y_train: Vec<f64> = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
        ];

        let dim: Vec<u32> = vec![20, 4];

        let epoch: u32 = 5;
        let c: f64 = 200.0;
        let tol: f64 = 0.001;

        let y_pred =
            transform(&x_train, &dim, &y_train, &x_train, &dim, c, epoch, tol);

        assert_eq!(y_pred.unwrap(), y_train);
    }
}
