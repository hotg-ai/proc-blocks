// use linfa_logistic::LogisticRegression;
use smartcore::metrics::{
    auc::AUC, f1::F1, mean_absolute_error::MeanAbsoluteError,
    mean_squared_error::MeanSquareError, precision::Precision, r2::R2,
    recall::Recall,
};

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

// Note: getrandom is pulled in by the linfa_logistic crate
getrandom::register_custom_getrandom!(unsupported_rng);

fn unsupported_rng(_buffer: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

/// A proc block which can perform linear regression
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Metric", env!("CARGO_PKG_VERSION"));
        metadata.set_description("for assessing prediction error");
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

        let f1 = TensorMetadata::new("accuracy");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        f1.add_hint(&hint);
        metadata.add_input(&f1);

        let f1 = TensorMetadata::new("f1_score");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        f1.add_hint(&hint);
        metadata.add_input(&f1);

        let precision = TensorMetadata::new("precision");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        precision.add_hint(&hint);
        metadata.add_input(&precision);

        let recall = TensorMetadata::new("recall");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        recall.add_hint(&hint);
        metadata.add_input(&recall);

        let auc = TensorMetadata::new("auc");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        auc.add_hint(&hint);
        metadata.add_input(&auc);

        let mae = TensorMetadata::new("mean_absolute_error");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        mae.add_hint(&hint);
        metadata.add_input(&mae);

        let mse = TensorMetadata::new("mean_square_error");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        mse.add_hint(&hint);
        metadata.add_input(&mse);

        let r2 = TensorMetadata::new("r2");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[1]));
        r2.add_hint(&hint);
        metadata.add_input(&r2);

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
            "f1_score",
            element_type,
            DimensionsParam::Fixed(&[1]),
        );

        ctx.add_output_tensor(
            "precision",
            element_type,
            DimensionsParam::Fixed(&[1]),
        );

        ctx.add_output_tensor(
            "recall",
            element_type,
            DimensionsParam::Fixed(&[1]),
        );

        ctx.add_output_tensor(
            "auc",
            element_type,
            DimensionsParam::Fixed(&[1]),
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

        ctx.add_output_tensor("r2", element_type, DimensionsParam::Fixed(&[1]));

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

        let y_pred = ctx.get_input_tensor("y_pred").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "y_pred".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let metric = transform(
            y_true.buffer.elements().to_vec(),
            y_pred.buffer.elements().to_vec(),
        );

        let f1 = vec![metric.0];

        ctx.set_output_tensor(
            "f1_score",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &f1.as_bytes(),
            },
        );

        let precision = vec![metric.1];

        ctx.set_output_tensor(
            "precision",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &precision.as_bytes(),
            },
        );

        let recall = vec![metric.2];

        ctx.set_output_tensor(
            "recall",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &recall.as_bytes(),
            },
        );

        let auc = vec![metric.3];

        ctx.set_output_tensor(
            "auc",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &auc.as_bytes(),
            },
        );

        let mae = vec![metric.4];

        ctx.set_output_tensor(
            "mean_absolute_error",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &mae.as_bytes(),
            },
        );

        let mse = vec![metric.5];

        ctx.set_output_tensor(
            "mean_square_error",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &mse.as_bytes(),
            },
        );

        let r2 = vec![metric.6];

        ctx.set_output_tensor(
            "r2",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[1 as u32],
                buffer: &r2.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let f1 = F1 { beta: 1.0 }.get_score(&y_pred, &y_true);
    let precision = Precision {}.get_score(&y_pred, &y_true);
    let recall = Recall {}.get_score(&y_pred, &y_true);
    let auc = AUC {}.get_score(&y_true, &y_pred);
    let mae = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
    let mse = MeanSquareError {}.get_score(&y_pred, &y_true);
    let r2 = R2 {}.get_score(&y_pred, &y_true);

    (f1, precision, recall, auc, mae, mse, r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_f1() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(0.5714285714285715, metric.0);
    }

    #[test]
    fn check_precision() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(0.6666666666666666, metric.1);
    }

    #[test]
    fn check_recall() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(0.5, metric.2);
    }

    #[test]
    fn check_auc() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(0.5, metric.3);
    }

    #[test]
    fn check_mae() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
        let metric = transform(y_true, y_pred);

        assert_eq!(0.5, metric.4);
    }

    #[test]
    fn check_mse() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(0.5, metric.5);
    }

    #[test]
    fn check_r2() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let metric = transform(y_true, y_pred);

        assert_eq!(-1.2499999999999996, metric.6);
    }
}
