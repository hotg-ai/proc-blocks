use smartcore::{
    linalg::naive::dense_matrix::*, linear::logistic_regression::*,
};

use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{ndarray, runtime_v1::*, BufferExt, SliceExt};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// A proc block which can perform linear regression
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        ensure_initialized();

        let metadata =
            Metadata::new("Logistic Regression", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "a linear approach for modelling the relationship between a scalar response and one or more explanatory variables",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("classification");
        metadata.add_tag("linear modeling");
        metadata.add_tag("analytics");

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
        ensure_initialized();

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
        ensure_initialized();

        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

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

fn transform(
    x_train: &[f64],
    x_train_dim: &[u32],
    y_train: &[f64],
    x_test: &[f64],
    x_test_dim: &[u32],
) -> Result<Vec<f64>, KernelError> {
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
    .map_err(|e| KernelError::Other(e.to_string()))?;

    let x_test = DenseMatrix::from_array(
        x_test_dim[0] as usize,
        x_test_dim[1] as usize,
        x_test,
    );

    lr.predict(&x_test)
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

        let y_pred = transform(&x_train, &dim, &y_train, &x_train, &dim);

        assert_eq!(y_pred.unwrap(), y_train);
    }

    #[test]
    #[should_panic]
    fn check_training_dimension_mismatch() {
        let x_train = vec![
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6,
            3.1, 1.5, 0.2, 5.0, 3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4,
            1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5,
            0.1, 7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5,
            5.5, 2.3, 4.0, 1.3, 6.5, 2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3,
            3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7,
            3.9, 1.4,
        ]; // shape [20, 4] after transformation
        let y_train: Vec<f64> = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.,
        ]; // shape[19, 1]

        let dim: Vec<u32> = vec![20, 4];

        let y_pred = transform(&x_train, &dim, &y_train, &x_train, &dim);

        assert_eq!(y_pred.unwrap(), y_train);
    }

    #[test]
    #[should_panic]
    fn check_dimension_mismatch() {
        let x_train = vec![
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6,
            3.1, 1.5, 0.2, 5.0, 3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4,
            1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5,
            0.1, 7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5,
            5.5, 2.3, 4.0, 1.3, 6.5, 2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3,
            3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7,
            3.9, 1.4,
        ];

        let x_test = vec![5.1, 3.5, 1.4, 0.2, 4.9]; // shape [1,5]: column has to be same as x_train

        let y_train: Vec<f64> = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
        ];

        let y_test: Vec<f64> = vec![0.];

        let dim: Vec<u32> = vec![20, 4];

        let y_pred = transform(&x_train, &dim, &y_train, &x_test, &vec![1, 5]);

        assert_eq!(y_pred.unwrap(), y_test);
    }
}
