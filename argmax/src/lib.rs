use crate::{
    proc_block_v1::{BadInputReason, GraphError, InvalidInput, KernelError},
    runtime_v1::*,
};
use hotg_rune_proc_blocks::BufferExt;
use std::cmp::Ordering;

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Arg Max", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("max");
        metadata.add_tag("index");
        metadata.add_tag("numeric");

        let input = TensorMetadata::new("input");
        let hint =
            supported_shapes(&[ElementType::F32], Dimensions::Fixed(&[0]));
        input.add_hint(&hint);
        metadata.add_input(&input);

        let max = TensorMetadata::new("max_index");
        max.set_description("The index of the element with the highest value");
        let hint =
            supported_shapes(&[ElementType::U32], Dimensions::Fixed(&[1]));
        max.add_hint(&hint);
        metadata.add_output(&max);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&id).ok_or_else(|| {
            GraphError::Other("Unable to get the graph context".to_string())
        })?;

        ctx.add_input_tensor(
            "input",
            ElementType::F32,
            Dimensions::Fixed(&[0]),
        );
        ctx.add_output_tensor(
            "max_index",
            ElementType::U32,
            Dimensions::Fixed(&[1]),
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

        let index = match element_type {
            ElementType::U8 => arg_max(buffer.elements::<u8>()),
            ElementType::I8 => arg_max(buffer.elements::<i8>()),
            ElementType::U16 => arg_max(buffer.elements::<u16>()),
            ElementType::I16 => arg_max(buffer.elements::<i16>()),
            ElementType::U32 => arg_max(buffer.elements::<u32>()),
            ElementType::I32 => arg_max(buffer.elements::<i32>()),
            ElementType::F32 => arg_max(buffer.elements::<f32>()),
            ElementType::U64 => arg_max(buffer.elements::<u64>()),
            ElementType::I64 => arg_max(buffer.elements::<i64>()),
            ElementType::F64 => arg_max(buffer.elements::<f64>()),
            other => {
                return Err(KernelError::Other(format!(
                "The Arg Max proc-block only accepts f32 tensors, found {:?}",
                other,
                )))
            },
        };

        let index = match index {
            Some(ix) => ix,
            None => {
                return Err(KernelError::Other(
                    "The input tensor was empty".to_string(),
                ))
            },
        };
        let resulting_tensor = (index as u32).to_le_bytes();

        ctx.set_output_tensor(
            "max_index",
            TensorParam {
                element_type,
                dimensions: &dimensions,
                buffer: &resulting_tensor,
            },
        );

        Ok(())
    }
}

fn arg_max<T>(values: &[T]) -> Option<usize>
where
    T: PartialOrd,
{
    let (index, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))?;

    Some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let values = [2.3, 12.4, 55.1, 15.4];

        let max = arg_max(&values).unwrap();

        assert_eq!(max, 2);
    }

    #[test]
    fn empty_inputs_are_an_error() {
        let empty: &[f32] = &[];
        let result = arg_max(empty);

        assert!(result.is_none());
    }
}
