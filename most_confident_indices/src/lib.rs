use std::{cmp::Ordering, convert::TryInto, fmt::Display};

use crate::{proc_block_v1::*, runtime_v1::*};

use hotg_rune_proc_blocks::{
    common, ndarray::ArrayView1, BufferExt, SliceExt, ValueType,
};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Most Confident Indices", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
                "Given some confidence values, create a tensor containing the indices of the top N highest confidences.",
            );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("classify");

        let count = ArgumentMetadata::new("count");
        count.set_description("The number of indices to return.");
        count.set_default_value("1");
        let hint =
            runtime_v1::supported_argument_type(ArgumentType::UnsignedInteger);
        count.add_hint(&hint);
        metadata.add_argument(&count);

        let element_type = ArgumentMetadata::new(common::element_type::NAME);
        element_type.set_description(common::element_type::DESCRIPTION);
        let hint = runtime_v1::interpret_as_string_in_enum(
            common::element_type::NUMERIC,
        );
        element_type.add_hint(&hint);
        metadata.add_argument(&element_type);

        let input = TensorMetadata::new("confidences");
        input.set_description("A 1D tensor of numeric confidence values.");
        let hint = supported_shapes(
            &[
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
            ],
            Dimensions::Dynamic,
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("indices");
        output
            .set_description("The indices, in order of descending confidence.");
        let hint =
            supported_shapes(&[ElementType::U32], Dimensions::Fixed(&[0]));
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        let element_type = get_element_type(|n| ctx.get_argument(n))
            .map_err(GraphError::InvalidArgument)?;
        let count = get_count(|n| ctx.get_argument(n))
            .map_err(GraphError::InvalidArgument)?;

        ctx.add_input_tensor("confidences", element_type, Dimensions::Dynamic);
        ctx.add_output_tensor(
            "indices",
            element_type,
            Dimensions::Fixed(&[count]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let count = get_count(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "indices".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        let count = count as usize;

        let indices = match element_type {
            ElementType::U8 => preprocess_buffer::<u8>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::I8 => preprocess_buffer::<i8>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::U16 => preprocess_buffer::<u16>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::I16 => preprocess_buffer::<i16>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::U32 => preprocess_buffer::<u32>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::I32 => preprocess_buffer::<i32>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::F32 => preprocess_buffer::<f32>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::U64 => preprocess_buffer::<u64>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::I64 => preprocess_buffer::<i64>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::F64 => preprocess_buffer::<f64>(&buffer, &dimensions)
                .and_then(|t| most_confident_indices(t, count))?,
            ElementType::Utf8 => {
                unreachable!("Already checked by get_element_type()")
            },
        };

        ctx.set_output_tensor(
            "indices",
            TensorParam {
                dimensions: &dimensions,
                element_type: ElementType::U32,
                buffer: indices.as_bytes(),
            },
        );

        Ok(())
    }
}

fn preprocess_buffer<'buf, T>(
    buffer: &'buf [u8],
    dimensions: &[u32],
) -> Result<ArrayView1<'buf, T>, KernelError>
where
    T: ValueType,
{
    buffer
        .view::<T>(dimensions)
        .and_then(|t| t.into_dimensionality())
        .map_err(|e| {
            KernelError::InvalidInput(InvalidInput::invalid_value(
                "confidences",
                e,
            ))
        })
}

fn most_confident_indices<T>(
    tensor: ArrayView1<T>,
    count: usize,
) -> Result<Vec<u32>, KernelError>
where
    T: PartialOrd + Copy,
{
    if count > tensor.len() {
        return Err(KernelError::InvalidArgument(
            InvalidArgument::invalid_value(
                "count",
                format!(
                    "Requesting {} indices from a tensor with only {} elements",
                    count,
                    tensor.len()
                ),
            ),
        ));
    }

    let mut indices_and_confidence: Vec<_> =
        tensor.iter().copied().enumerate().collect();

    indices_and_confidence
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));

    Ok(indices_and_confidence
        .into_iter()
        .map(|(index, _confidence)| index.try_into().unwrap())
        .take(count)
        .collect())
}

fn get_count(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<u32, InvalidArgument> {
    get_argument("count")
        .ok_or_else(|| InvalidArgument::not_found("count"))?
        .parse::<u32>()
        .map_err(|e| InvalidArgument::invalid_value("count", e))
}

fn get_element_type(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<ElementType, InvalidArgument> {
    match get_argument("element_type").as_deref() {
        Some("u8") => Ok(ElementType::U8),
        Some("i8") => Ok(ElementType::I8),
        Some("u16") => Ok(ElementType::U16),
        Some("i16") => Ok(ElementType::I16),
        Some("u32") => Ok(ElementType::U32),
        Some("i32") => Ok(ElementType::I32),
        Some("f32") => Ok(ElementType::F32),
        Some("u64") => Ok(ElementType::U64),
        Some("i64") => Ok(ElementType::I64),
        Some("f64") => Ok(ElementType::F64),
        Some(other) => Err(InvalidArgument::invalid_value(
            "element_type",
            format!("Unsupported element type: {}", other),
        )),
        None => Err(InvalidArgument::not_found("element_type")),
    }
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

impl InvalidInput {
    fn invalid_value(name: impl Into<String>, reason: impl Display) -> Self {
        InvalidInput {
            name: name.into(),
            reason: BadInputReason::InvalidValue(reason.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

    #[test]
    fn only_works_with_1d() {
        let buffer = [1, 2, 3, 4, 5, 6];

        let error = preprocess_buffer::<u8>(&buffer, &[2, 3]).unwrap_err();

        assert!(matches!(error, KernelError::InvalidInput(_)));
    }

    #[test]
    fn tensors_equivalent_to_1d_are_okay_too() {
        let buffer = [1, 2, 3, 4, 5, 6];

        let error = preprocess_buffer::<u8>(&buffer, &[1, 6, 1]).unwrap_err();

        assert!(matches!(error, KernelError::InvalidInput(_)));
    }

    #[test]
    fn count_must_be_less_than_input_size() {
        let input = ndarray::arr1(&[1_u32, 2, 3]);

        let error = most_confident_indices(input.view(), 42).unwrap_err();

        assert!(matches!(error, KernelError::InvalidArgument(_)));
    }

    #[test]
    fn get_top_3_values() {
        let elements = ndarray::arr1(&[0.0, 0.5, 10.0, 3.5, -200.0]);

        let got = most_confident_indices(elements.view(), 3).unwrap();

        assert_eq!(got, &[2, 3, 1]);
    }
}
