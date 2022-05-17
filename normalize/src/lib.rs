use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};
use num_traits::ToPrimitive;

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// Normalize the input to the range `[0, 1]`.
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Normalize", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "Normalize a tensor's elements to the range, `[0, 1]`.",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("normalize");

        let input = TensorMetadata::new("input");
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
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("normalized");
        output.set_description("normalized tensor in the range [0, 1]");
        let hint =
            supported_shapes(&[ElementType::F32], DimensionsParam::Dynamic);
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

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

        ctx.add_input_tensor("input", element_type, DimensionsParam::Dynamic);
        ctx.add_output_tensor(
            "normalized",
            ElementType::F32,
            DimensionsParam::Dynamic,
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

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

        let output = match element_type {
            ElementType::U8 => transform(buffer.elements::<u8>()),
            ElementType::I8 => transform(buffer.elements::<i8>()),
            ElementType::U16 => transform(buffer.elements::<u16>()),
            ElementType::I16 => transform(buffer.elements::<i16>()),
            ElementType::U32 => transform(buffer.elements::<u32>()),
            ElementType::I32 => transform(buffer.elements::<i32>()),
            ElementType::F32 => transform(buffer.elements::<f32>()),
            ElementType::U64 => transform(buffer.elements::<u64>()),
            ElementType::I64 => transform(buffer.elements::<i64>()),
            ElementType::F64 => transform(buffer.elements::<f64>()),
            other => {
                return Err(KernelError::Other(format!(
                "The Normalize proc-block doesn't support {:?} element type",
                other,
                )))
            },
        };

        let output = match output {
            Some(out) => out,
            None => {
                return Err(KernelError::Other(
                    "The input tensor was empty".to_string(),
                ))
            },
        };

        ctx.set_output_tensor(
            "normalized",
            TensorParam {
                element_type: ElementType::U32,
                dimensions: &dimensions,
                buffer: &output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn transform<T>(input: &[T]) -> Option<Vec<f32>>
where
    T: ToPrimitive,
{
    let (min, max) =
        min_max(input.iter().map(|e| e.to_f32().unwrap())).unwrap();
    let range = max - min;
    if range == 0.0 {
        return Some(vec![0.0; input.len()]);
    }
    let mut v: Vec<f32> = Vec::new();

    for e in input {
        let e = e.to_f32().unwrap();
        v.push((e - min) / range)
    }
    return Some(v);
}

fn min_max(items: impl Iterator<Item = f32>) -> Option<(f32, f32)> {
    items.into_iter().fold(None, |bounds, item| match bounds {
        Some((min, max)) => {
            let min = if item < min { item } else { min };
            let max = if max < item { item } else { max };
            Some((min, max))
        },
        None => Some((item, item)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let input = [0.0, 1.0, 2.0];

        let output = transform(&input).unwrap();

        assert_eq!(output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn it_works_with_integers() {
        let input = [0, 1, 2];

        let output = transform(&input).unwrap();

        assert_eq!(output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn handle_empty() {
        let input = [0.0; 384];

        let output = transform(&input.clone()).unwrap();

        assert_eq!(output, input);
        assert_eq!(output.len(), 384);
    }
}
