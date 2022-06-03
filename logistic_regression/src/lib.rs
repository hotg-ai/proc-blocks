use linfa::{
    traits::{Fit, Predict},DatasetView,
};
use linfa_logistic::LogisticRegression;

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    ndarray::{Array1,  ArrayView1,
        ArrayView2,
    },
    runtime_v1::*,
    BufferExt, SliceExt,
};


wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

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
        let hint = supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_train.add_hint(&hint);
        metadata.add_input(&x_train);

        let y_train = TensorMetadata::new("y_train");
        let hint =
            supported_shapes(&[ElementType::I32], DimensionsParam::Fixed(&[0]));
        y_train.add_hint(&hint);
        metadata.add_input(&y_train);

        let x_test = TensorMetadata::new("x_test");
        let hint = supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
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

        ctx.add_input_tensor("x_train", element_type, DimensionsParam::Fixed(&[0, 0]));

        ctx.add_input_tensor(
            "y_train",
            ElementType::I32,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_input_tensor("x_test", element_type, DimensionsParam::Fixed(&[0, 0]));

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

        let y_test_dimension = [x_train.dimensions[0], 1];

        let y_train = y_train
            .buffer
            .view::<i32>(&y_train.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "bytes".to_string(),
                    reason: BadInputReason::InvalidValue(e.to_string()),
                })
            })?;

        let x_train = x_train
            .buffer
            .view::<f64>(&x_train.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "bytes".to_string(),
                    reason: BadInputReason::InvalidValue(e.to_string()),
                })
            })?;

        let x_test = x_test
            .buffer
            .view::<f64>(&x_test.dimensions)
            .and_then(|t| t.into_dimensionality())
            .map_err(|e| {
                KernelError::InvalidInput(InvalidInput {
                    name: "bytes".to_string(),
                    reason: BadInputReason::InvalidValue(e.to_string()),
                })
            })?;

        let output = transform(x_train, y_train, x_test);

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

fn transform<'a>(
    x_train: ArrayView2<'a, f64>,
    y_train: ArrayView1<'a, i32>,
    x_test: ArrayView2<'_, f64>,
) -> Array1<i32> {
    let training_data = DatasetView::new(x_train, y_train);

    let model = LogisticRegression::default().fit(&training_data).unwrap();
    let prediction = model.predict(&x_test);

    prediction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_model(){
        let dataset = linfa_datasets::winequality().map_targets(|x| if *x > 6 { 0 } else { 1 });
        let (train, valid) = dataset.split_with_ratio(0.8);
        let y_train = train.targets();
        let x_train = train.records();
        let x_test = valid.records();
        let y_test = valid.targets();
        let y_pred = transform(x_train.view(), y_train.view(), x_test.view());

        println!("{:?}", y_pred);

        assert_eq!(y_pred, y_test);
        
    }
}
