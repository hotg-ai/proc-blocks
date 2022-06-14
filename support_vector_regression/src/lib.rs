use std::{convert::TryInto, fmt::Display, str::FromStr};

use smartcore::{
    linalg::naive::dense_matrix::*,
    svm::{
        svr::{SVRParameters, SVR},
        Kernels,
    },
};

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    runtime_v1::{self, *},
    BufferExt, SliceExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

// Note: getrandom is pulled in by the linfa_logistic crate
getrandom::register_custom_getrandom!(unsupported_rng);

fn unsupported_rng(_buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

/// a binary classifier that uses an optimal hyperplane to separate the points
/// in the input variable space by their class.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new(
            "Support Vector Regression",
            env!("CARGO_PKG_VERSION"),
        );
        metadata.set_description(
            "a binary approach for modelling the relationship between a scalar response and one or more explanatory variables",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("regression");
        metadata.add_tag("analytics");

        let eps = ArgumentMetadata::new("eps");
        eps.set_description("epsilon");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        eps.add_hint(&hint);
        eps.set_default_value("2.0");
        metadata.add_argument(&eps);

        let c = ArgumentMetadata::new("c");
        c.set_description("Penalizing parameter");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        c.add_hint(&hint);
        c.set_default_value("10.0");
        metadata.add_argument(&c);

        let tol = ArgumentMetadata::new("tolerance");
        tol.set_description("Tolerance for stopping criterion");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        tol.add_hint(&hint);
        tol.set_default_value("0.001");
        metadata.add_argument(&tol);

        let element_type = ArgumentMetadata::new("element_type");
        element_type
            .set_description("The type of tensor this proc-block will accept");
        element_type.set_default_value("f64");
        element_type.add_hint(&interpret_as_string_in_enum(&[
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        ]));
        metadata.add_argument(&element_type);

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

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("f64") => ElementType::F64,
            Some(_) => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::InvalidValue(
                        "Unsupported element type".to_string(),
                    ),
                }));
            },
            None => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::NotFound,
                }))
            },
        };

        ctx.add_input_tensor(
            "x_train",
            element_type,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_input_tensor(
            "y_train",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "x_test",
            element_type,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_output_tensor(
            "y_test",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let eps: f64 = get_args("eps", |n| ctx.get_argument(n))
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

        let y_train = ctx.get_input_tensor("y_train").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "y_train".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let x_test = ctx.get_input_tensor("x_test").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "x_test".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let output = transform(
            &x_train.buffer.elements(),
            &x_train.dimensions,
            &y_train.buffer.elements(),
            &x_test.buffer.elements(),
            &x_test.dimensions,
            c,
            eps,
            tol,
        );

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
    eps: f64,
    tol: f64,
) -> Vec<f64> {
    // todo: let user change the kernel. Right now setting it to 'linear'
    let svc_parameters = SVRParameters::default()
        .with_c(c)
        .with_eps(eps.try_into().unwrap())
        .with_kernel(Kernels::linear())
        .with_tol(tol);

    let x_train = DenseMatrix::from_array(
        x_train_dim[0] as usize,
        x_train_dim[1] as usize,
        x_train,
    );

    let model = SVR::fit(&x_train, &y_train.to_vec(), svc_parameters).unwrap();

    let x_test = DenseMatrix::from_array(
        x_test_dim[0] as usize,
        x_test_dim[1] as usize,
        x_test,
    );

    let y_hat = model.predict(&x_test).unwrap();

    y_hat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_model() {
        let x_train = [
            234.289, 235.6, 159.0, 107.608, 1947., 60.323, 259.426, 232.5,
            145.6, 108.632, 1948., 61.122, 258.054, 368.2, 161.6, 109.773,
            1949., 60.171, 284.599, 335.1, 165.0, 110.929, 1950., 61.187,
            328.975, 209.9, 309.9, 112.075, 1951., 63.221, 346.999, 193.2,
            359.4, 113.270, 1952., 63.639, 365.385, 187.0, 354.7, 115.094,
            1953., 64.989, 363.112, 357.8, 335.0, 116.219, 1954., 63.761,
            397.469, 290.4, 304.8, 117.388, 1955., 66.019, 419.180, 282.2,
            285.7, 118.734, 1956., 67.857, 442.769, 293.6, 279.8, 120.445,
            1957., 68.169, 444.546, 468.1, 263.7, 121.950, 1958., 66.513,
            482.704, 381.3, 255.2, 123.366, 1959., 68.655, 502.601, 393.1,
            251.4, 125.368, 1960., 69.564, 518.173, 480.6, 257.2, 127.852,
            1961., 69.331, 554.894, 400.7, 282.7, 130.081, 1962., 70.551,
        ];

        let y_train: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6,
            108.4, 110.8, 112.6, 114.2, 115.7, 116.9,
        ];

        let dim: Vec<u32> = vec![16, 6];

        let y_pred = transform(
            &x_train, &dim, &y_train, &x_train, &dim, 10.0, 2.0, 0.001,
        );

        println!("{:?}", y_pred);

        let should_be = vec![
            85.00037818041841,
            86.75542812311954,
            89.1978358812151,
            90.98812129438727,
            96.13994481889046,
            98.56353286481169,
            99.91360351464635,
            101.99962181958176,
            103.10761964972573,
            104.36416760001185,
            106.40037818041844,
            108.97089143261519,
            110.59974385982332,
            112.38558374212687,
            115.24619508029843,
            117.6680182728901,
        ];
        assert_eq!(y_pred, should_be);
    }
}
