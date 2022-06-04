// use linfa_logistic::LogisticRegression;
use smartcore::{
    linalg::naive::dense_matrix::*, linear::logistic_regression::*,
};

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    runtime_v1::*,
    BufferExt, SliceExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

// Note: getrandom is pulled in by the linfa_logistic crate
getrandom::register_custom_getrandom!(unsupported_rng);

fn unsupported_rng(_buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

/// A proc block which can parse a string to numbers.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Logistic Regression", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "a statistical model that models the probability of one event taking place by having the log-odds for the event be a linear combination of one or more independent variables.",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("regression");
        metadata.add_tag("binary classification");
        metadata.add_tag("categorical dependent variable");

        let x_train = TensorMetadata::new("x_train");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_train.add_hint(&hint);
        metadata.add_input(&x_train);

        let y_train = TensorMetadata::new("y_train");
        let hint =
            supported_shapes(&[ElementType::I32], DimensionsParam::Fixed(&[0]));
        y_train.add_hint(&hint);
        metadata.add_input(&y_train);

        let x_test = TensorMetadata::new("x_test");
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_test.add_hint(&hint);
        metadata.add_input(&x_test);

        let y_test = TensorMetadata::new("y_test");
        let supported_types = [ElementType::I32];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y_test.add_hint(&hint);
        metadata.add_output(&y_test);
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
            ElementType::I32,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "x_test",
            element_type,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_output_tensor(
            "y_test",
            ElementType::I32,
            DimensionsParam::Fixed(&[0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

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
        );

        let y_test_dimension = [x_test.dimensions[0]];

        ctx.set_output_tensor(
            "y_test",
            TensorParam {
                element_type: ElementType::I32,
                dimensions: &y_test_dimension,
                buffer: &output.to_vec().as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(
    x_train: &[f64],
    x_train_dim: &[u32],
    y_train: &[f64],
    x_test: &[f64],
    x_test_dim: &[u32],
) -> Vec<f64> {
    // Iris data
    let x_train = DenseMatrix::from_array(
        x_train_dim[0] as usize,
        x_train_dim[1] as usize,
        x_train,
    );

    let lr = LogisticRegression::fit(
        &x_train,
        &y_train.to_vec(),
        Default::default(),
    )
    .unwrap();

    let x_test = DenseMatrix::from_array(
        x_test_dim[0] as usize,
        x_test_dim[1] as usize,
        x_test,
    );

    let y_hat = lr.predict(&x_test).unwrap();

    y_hat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_model() {
        let x_train =
            [5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4];
        let y_train: Vec<f64> = vec![0., 0., 1.];

        let dim: Vec<u32> = vec![3, 4];

        let y_pred = transform(
            &x_train,
            &dim,
            &y_train,
            &x_train,
            &dim,
        );

        println!("{:?}", y_pred);

        assert_eq!(y_pred, y_train);
    }
}
