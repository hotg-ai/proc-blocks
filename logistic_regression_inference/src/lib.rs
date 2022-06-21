use hotg_rune_proc_blocks::{guest::{
    parse, Argument, ArgumentMetadata, ArgumentType, CreateError, Metadata, ProcBlock, RunError, Tensor, TensorConstraint,
    TensorConstraints, TensorMetadata, ElementTypeConstraint,
}, ndarray::Array1};
use std::{fmt::Debug, str::FromStr};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: LogisticInference,
}

use serde_json;

use smartcore::
    {linear::logistic_regression::LogisticRegression,linalg::Matrix, math::num::RealNumber};

fn metadata() -> Metadata {
    Metadata::new("Logistic Regression Inference", env!("CARGO_PKG_VERSION"))
        .with_description(
                "a json file which contains serialized model",
            )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("classify")
        .with_argument(ArgumentMetadata::new("model_file")
        .with_hint(ArgumentType::String)
        .with_default_value("")
    )
    .with_input(TensorMetadata::new("x_test").with_description("test samples"))
    .with_output(TensorMetadata::new("y_pred").with_description("predicted labels"))
}

#[derive(Debug, Clone, PartialEq)]
struct LogisticInference {
    model_file: String,
}

impl ProcBlock for LogisticInference {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "x_test", ElementTypeConstraint::F64, vec![0, 0])],
            outputs: vec![TensorConstraint::new(
                "y_pred", ElementTypeConstraint::F64, vec![0])],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let x_test = Tensor::get_named(&inputs, "x_test")?.view::<u32>()?;
        let model: LogisticRegression= serde_json::from_str(&self.model_file).map_err(RunError::other);

        let y_pred = model
        .predict(&x_test)
        .map(Array1::from_vec)
        .map_err(RunError::other);
        
        Ok(vec![Tensor::new_1d("y_pred", &y_pred)])
    }
}


impl TryFrom<Vec<Argument>> for LogisticInference {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let model_file = parse::required_arg(&args, "model_file")?;

        Ok(LogisticInference { model_file })
    }
}
