use std::{fmt::Display, str::FromStr};

use smartcore::{
    linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix},
    model_selection::train_test_split,
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

/// A proc block which can perform linear regression
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Train-Test-Split", env!("CARGO_PKG_VERSION"));
        metadata.set_description("a random split into training and test sets");
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("split");
        metadata.add_tag("data processing");
        metadata.add_tag("analytics");

        let x = TensorMetadata::new("features");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x.add_hint(&hint);
        metadata.add_input(&x);

        // todo: have to make it dynamic size because y could be 1-d or 2-d
        let y = TensorMetadata::new("targets");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0]));
        y.add_hint(&hint);
        metadata.add_input(&y);

        let test_size = ArgumentMetadata::new("test_size");
        test_size.set_description(
            "the proportion of the dataset to include in the test split",
        );
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        test_size.add_hint(&hint);
        test_size.set_default_value("0.2");
        metadata.add_argument(&test_size);

        let element_type = ArgumentMetadata::new("element_type");
        element_type
            .set_description("The type of tensor this proc-block will accept");
        element_type.set_default_value("f64");
        element_type.add_hint(&runtime_v1::interpret_as_string_in_enum(&[
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        ]));
        metadata.add_argument(&element_type);

        let x_train = TensorMetadata::new("x_train");
        let supported_types = [ElementType::F64];
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_train.add_hint(&hint);
        metadata.add_output(&x_train);

        let y_train = TensorMetadata::new("y_train");
        let hint =
            supported_shapes(&[ElementType::F64], DimensionsParam::Fixed(&[0]));
        y_train.add_hint(&hint);
        metadata.add_output(&y_train);

        let x_test = TensorMetadata::new("x_test");
        let hint =
            supported_shapes(&supported_types, DimensionsParam::Fixed(&[0, 0]));
        x_test.add_hint(&hint);
        metadata.add_output(&x_test);

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
            "features",
            element_type,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_input_tensor(
            "targets",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
            "x_train",
            element_type,
            DimensionsParam::Fixed(&[0, 0]),
        );

        ctx.add_output_tensor(
            "y_train",
            element_type,
            DimensionsParam::Fixed(&[0]),
        );

        ctx.add_output_tensor(
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

        let x = ctx.get_input_tensor("features").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "features".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let y = ctx.get_input_tensor("targets").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "targets".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let test_size: f32 = get_args("test_size", |n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let (x_train, x_test, y_train, y_test, train_dim, test_dim) = transform(
            &x.buffer.elements(),
            &x.dimensions,
            (&y.buffer.elements()).to_vec(),
            test_size,
        );

        ctx.set_output_tensor(
            "x_train",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[train_dim.0 as u32, train_dim.1 as u32],
                buffer: &x_train.as_bytes(),
            },
        );

        ctx.set_output_tensor(
            "x_test",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[test_dim.0 as u32, test_dim.1 as u32],
                buffer: &x_test.as_bytes(),
            },
        );

        ctx.set_output_tensor(
            "y_train",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[train_dim.0 as u32],
                buffer: &y_train.as_bytes(),
            },
        );

        ctx.set_output_tensor(
            "y_test",
            TensorParam {
                element_type: ElementType::F64,
                dimensions: &[test_dim.0 as u32],
                buffer: &y_test.as_bytes(),
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
    x: &[f64],
    x_dim: &[u32],
    y: Vec<f64>,
    test_size: f32,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    (usize, usize),
    (usize, usize),
) {
    let x = DenseMatrix::from_array(x_dim[0] as usize, x_dim[1] as usize, x);

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, test_size, false);
    let train_dim = x_train.shape();
    let test_dim = x_test.shape();
    let x_train: Vec<f64> = x_train.iter().map(|f| f).collect();
    let x_test: Vec<f64> = x_test.iter().map(|f| f).collect();

    (x_train, x_test, y_train, y_test, train_dim, test_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_test_dim() {
        let x = [
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4, 5.1,
            3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4,
        ];
        let y: Vec<f64> = vec![0., 0., 1., 0., 0., 1.];

        let dim: Vec<u32> = vec![6, 4];

        let (_x_train, _x_test, _y_train, _y_test, _train_dim, test_dim) =
            transform(&x, &dim, y, 0.2);

        let should_be = (1, 4);

        assert_eq!(test_dim, should_be);
    }
    #[test]
    fn check_train_dim() {
        let x = [
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4, 5.1,
            3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 5.2, 2.7, 3.9, 1.4,
        ];
        let y: Vec<f64> = vec![0., 0., 1., 0., 0., 1.];

        let dim: Vec<u32> = vec![6, 4];

        let (_x_train, _x_test, _y_train, _y_test, train_dim, _test_dim) =
            transform(&x, &dim, y, 0.2);

        let should_be = (5, 4);
        assert_eq!(train_dim, should_be);
    }
}
