// use linfa_logistic::LogisticRegression;
use smartcore::metrics::*;

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt, ndarray};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// A proc block which can perform linear regression
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Accuracy", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "calculates accuracy of predicted labels when compared to true labels",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("metric");
        metadata.add_tag("analytics");

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

        let accuracy = TensorMetadata::new("accuracy");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        accuracy.add_hint(&hint);
        metadata.add_output(&accuracy);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "y_true",
            ElementType::F64,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor(
            "y_pred",
            ElementType::F64,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
            "accuracy",
            ElementType::F64,
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
        let _ypred: ndarray::ArrayView1<f64> = y_pred
            .buffer
            .view(&y_pred.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "y_pred".to_string(),
                    reason: BadInputReason::Other(e.to_string()),
                })
            })?;

        if y_true.element_type != ElementType::F64
            || y_pred.element_type != ElementType::F64
        {
            return Err(KernelError::Other(format!(
                "This proc-block only support f64 element type",
            )));
        }

        let accuracy = transform(
            y_true.buffer.elements().to_vec(),
            y_pred.buffer.elements().to_vec(),
        )
        .unwrap();

        let output = vec![accuracy];

        ctx.set_output_tensor(
            "accuracy",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(y_true: Vec<f64>, y_pred: Vec<f64>) -> Result<f64, KernelError> {
    if y_true.len() != y_pred.len() {
        return Err( KernelError::Other(format!(
        "Dimension Mismatch: dimension of true labels is {} while {} for predicted labels", y_true.len(), y_pred.len()
    )));
    }
    Ok(ClassificationMetrics::accuracy().get_score(&y_true, &y_pred))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_transform() {
        let y_pred: Vec<f64> = vec![0., 2., 1., 3.];
        let y_true: Vec<f64> = vec![0., 1., 2., 3.];

        let accuracy = transform(y_true, y_pred);

        assert_eq!(0.5, accuracy.unwrap());
    }
}
