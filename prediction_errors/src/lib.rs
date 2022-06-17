use smartcore::metrics::{
    mean_absolute_error::MeanAbsoluteError, mean_squared_error::MeanSquareError,
};

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{ndarray, runtime_v1::*, BufferExt, SliceExt};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// a proc-block to find Mean Absolute Error and Mean Squared Error
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Errors", env!("CARGO_PKG_VERSION"));
        metadata.set_description("for assessing prediction error");
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("metric");
        metadata.add_tag("analytics");
        metadata.add_tag("loss");

        let y_true = TensorMetadata::new("y_true");
        let hint =
            supported_shapes(&[ElementType::F64], DimensionsParam::Fixed(&[0]));
        y_true.add_hint(&hint);
        metadata.add_input(&y_true);

        let y_pred = TensorMetadata::new("y_pred");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y_pred.add_hint(&hint);
        metadata.add_input(&y_pred);

        let mae = TensorMetadata::new("mean_absolute_error");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        mae.add_hint(&hint);
        metadata.add_output(&mae);

        let mse = TensorMetadata::new("mean_square_error");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        mse.add_hint(&hint);
        metadata.add_output(&mse);

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
            "y_true",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "y_pred",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
            "mean_absolute_error",
            element_type,
            DimensionsParam::Fixed(&[1]),
        );

        ctx.add_output_tensor(
            "mean_square_error",
            element_type,
            DimensionsParam::Fixed(&[1]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let y_true = ctx.get_input_tensor("y_true").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "y_true".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;
        let _ytrue: ndarray::ArrayView1<f64> = y_true
            .buffer
            .view(&y_true.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "y_train".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

        let y_pred = ctx.get_input_tensor("y_pred").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "y_pred".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;
        let _ypred: ndarray::ArrayView1<f64> = y_true
            .buffer
            .view(&y_pred.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "y_train".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

            

        let metric = transform(
            y_true.buffer.elements().to_vec(),
            y_pred.buffer.elements().to_vec(),
        )
        .unwrap();

        let mae = vec![metric.0];

        ctx.set_output_tensor(
            "mean_absolute_error",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &mae.as_bytes(),
            },
        );

        let mse = vec![metric.1];

        ctx.set_output_tensor(
            "mean_square_error",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &mse.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
) -> Result<(f64, f64), KernelError> {
    if y_true.len() != y_pred.len() {
        return Err( KernelError::Other(format!(
        "Dimension Mismatch: dimension of true labels is {} while {} for predicted labels", y_true.len(), y_pred.len()
    )));
    }

    let mae = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
    let mse = MeanSquareError {}.get_score(&y_pred, &y_true);

    Ok((mae, mse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_mae() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
        let metric = transform(y_true, y_pred).unwrap();

        assert_eq!(0.5, metric.0);
    }

    #[test]
    fn check_mse() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred).unwrap();

        assert_eq!(0.5, metric.1);
    }

    #[test]
    #[should_panic]
    fn dimension_mismatch() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1.];

        let metric = transform(y_true, y_pred).unwrap();

        assert_eq!(0.5, metric.1);
    }
}
