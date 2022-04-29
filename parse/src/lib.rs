use crate::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError,
    },
    runtime_v1::*,
};
use hotg_rune_proc_blocks::{common, BufferExt, SliceExt};
use std::{fmt::Display, str::FromStr};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// A proc block which can parse a string to numbers.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Parse", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("string");
        metadata.add_tag("numbers");

        let input = TensorMetadata::new("input_string_of_numbers");
        let hint =
            supported_shapes(&[ElementType::Utf8], DimensionsParam::Dynamic);
        input.add_hint(&hint);
        metadata.add_input(&input);

        let element_type = ArgumentMetadata::new(common::element_type::NAME);
        element_type.set_description("The type that values get parsed into");
        element_type.add_hint(&runtime_v1::interpret_as_string_in_enum(
            common::element_type::NUMERIC,
        ));
        metadata.add_argument(&element_type);

        let output = TensorMetadata::new("parsed_numbers");
        output.set_description("The parsed values");
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
        let hint = supported_shapes(&supported_types, DimensionsParam::Dynamic);
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id).ok_or_else(|| {
            GraphError::Other("Unable to get the graph context".to_string())
        })?;

        ctx.add_input_tensor(
            "input_string_of_numbers",
            ElementType::Utf8,
            DimensionsParam::Dynamic,
        );

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("u8") => ElementType::U8,
            Some("i8") => ElementType::I8,
            Some("u16") => ElementType::U16,
            Some("i16") => ElementType::I16,
            Some("u32") => ElementType::U32,
            Some("i32") => ElementType::I32,
            Some("f32") => ElementType::F32,
            Some("u64") => ElementType::U64,
            Some("i64") => ElementType::I64,
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

        ctx.add_output_tensor(
            "parsed_numbers",
            element_type,
            DimensionsParam::Dynamic,
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).ok_or_else(|| {
            KernelError::Other("Unable to get the kernel context".to_string())
        })?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "input".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let numbers = match element_type {
            ElementType::Utf8 => buffer
                .strings()
                .map_err(|e| KernelError::Other(e.to_string()))?,
            other => {
                return Err(KernelError::Other(format!(
                "The Parse proc-block only accepts Utf8 tensors, found {:?}",
                other,
                )))
            },
        };

        match ctx.get_argument("element_type").as_deref() {
            Some("u8") => {
                let transformed = transform::<u8>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::U8,
                    dimensions: &dimensions,
                    buffer: &transformed,
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("i8") => {
                let transformed = transform::<i8>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::I8,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("u16") => {
                let transformed = transform::<u16>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::U16,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("i16") => {
                let transformed = transform::<i16>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::I16,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("u32") => {
                let transformed = transform::<u32>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::U32,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("i32") => {
                let transformed = transform::<i32>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::I32,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("f32") => {
                let transformed = transform::<f32>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::F32,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("u64") => {
                let transformed = transform::<u64>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::U64,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("i64") => {
                let transformed = transform::<i64>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::I64,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some("f64") => {
                let transformed = transform::<f64>(&numbers)?;
                let output = TensorParam {
                    element_type: ElementType::F64,
                    dimensions: &dimensions,
                    buffer: transformed.as_bytes(),
                };
                ctx.set_output_tensor("parsed_numbers", output);
            },
            Some(_) => {
                return Err(KernelError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::InvalidValue(
                        "Unsupported element type".to_string(),
                    ),
                }));
            },
            None => {
                return Err(KernelError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::NotFound,
                }));
            },
        }

        Ok(())
    }
}

fn transform<T>(inputs: &[&str]) -> Result<Vec<T>, KernelError>
where
    T: FromStr,
    T::Err: Display,
{
    let mut values: Vec<T> = Vec::new();

    for input in inputs {
        match input.parse() {
            Ok(v) => values.push(v),
            Err(e) => {
                return Err(KernelError::Other(format!(
                    "Unable to parse \"{input}\" because of {e}"
                )))
            },
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;

    #[test]
    fn test_for_number_in_vec() {
        let bytes = vec!["5", "6", "7"];
        let output: Vec<i32> = transform(&bytes).unwrap();
        let should_be = vec![5, 6, 7];
        assert_eq!(output, should_be);
    }

    #[test]
    fn test_for_invalid_data_type() {
        let bytes = ["1.0", "a"];
        let err = transform::<f32>(&bytes).unwrap_err();

        match err {
            KernelError::Other(msg) => assert_eq!(
                msg,
                "Unable to parse \"a\" because of invalid float literal"
            ),
            other => panic!("Unexpected error: {:?}", other),
        }
    }
}
