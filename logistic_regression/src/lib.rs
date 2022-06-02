use linfa::traits::{Fit, Predict};
use linfa_datasets::winequality;
use linfa_logistic::LogisticRegression;

use crate::proc_block_v1::{
    BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
    InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    common,
    runtime_v1::{self, *},
    BufferExt, SliceExt,
};
use std::{fmt::Display, str::FromStr};

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
        let supported_types = 
            [ElementType::F32]
        ;
        let hint = supported_shapes(&supported_types, DimensionsParam::Dynamic);
        x_train.add_hint(&hint);
        metadata.add_input(&x_train);

        let y_train = TensorMetadata::new("y_train");
        let hint = supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y_train.add_hint(&hint);
        metadata.add_input(&y_train);

        let x_test = TensorMetadata::new("x_test");
        let hint = supported_shapes(&supported_types, DimensionsParam::Dynamic);
        x_test.add_hint(&hint);
        metadata.add_input(&x_test);

        let y_test = TensorMetadata::new("y_test");
        let supported_types = [
            ElementType::U8,
            ElementType::I8,
            ElementType::U16,
            ElementType::I16,
            ElementType::U32,
            ElementType::I32,
            ElementType::F32,
            ElementType::U64,
            ElementType::I64,
            ElementType::F64,
        ];
        let hint = supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y_test.add_hint(&hint);
        metadata.add_output(&y_test);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("f32") => ElementType::F32,
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

        ctx.add_input_tensor("x_train", element_type, DimensionsParam::Dynamic);

        ctx.add_input_tensor("y_train", element_type, DimensionsParam::Fixed(&[0]));

        ctx.add_input_tensor("x_test", element_type, DimensionsParam::Dynamic);

        ctx.add_output_tensor("y_test", element_type, DimensionsParam::Fixed(&[0]));

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
            x_train.buffer.view::<f32>(&x_train.dimensions).unwrap(),
            y_train.buffer.view::<f32>(&y_train.dimensions).unwrap(),
            x_test.buffer.view::<f32>(&x_test.dimensions).unwrap(),
        );

        Ok(())
    }
}

fn transform(x_train: &[f32], y_train: &[f32], x_test: &[f32]) -> Vec<f32> {
    todo!()
}
