use crate::proc_block_v2::*;
use hotg_rune_proc_blocks::{ndarray, BufferExt, SliceExt};
use std::cmp::Ordering;
use wit_bindgen_rust::Handle;

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v2.wit");

hotg_rune_proc_blocks::generate_support!(crate::proc_block_v2);

struct ProcBlockV2;

impl proc_block_v2::ProcBlockV2 for ProcBlockV2 {
    fn metadata() -> Metadata {
        Metadata {
            name: "Arg Max".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            tags: vec![
                "max".to_string(),
                "index".to_string(),
                "numeric".to_string(),
            ],
            description: Some(env!("CARGO_PKG_DESCRIPTION").to_string()),
            repository: Some(env!("CARGO_PKG_REPOSITORY").to_string()),
            homepage: Some(env!("CARGO_PKG_HOMEPAGE").to_string()),
            arguments: Vec::new(),
            inputs: vec![TensorMetadata {
                name: "input".to_string(),
                description: None,
                hints: Vec::new(),
            }],
            outputs: vec![TensorMetadata {
                name: "max_index".to_string(),
                description: Some(
                    "The index of the element with the highest value"
                        .to_string(),
                ),
                hints: Vec::new(),
            }],
        }
    }
}

pub struct Node;

impl proc_block_v2::Node for Node {
    fn new(_args: Vec<Argument>) -> Result<Handle<Self>, ArgumentError> {
        Ok(Handle::new(Node))
    }

    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint {
                name: "input".to_string(),
                element_type: !ElementTypeConstraint::UTF8,
                dimensions: Dimensions::Fixed(vec![0]),
            }],
            outputs: vec![TensorConstraint {
                name: "max_index".to_string(),
                element_type: ElementTypeConstraint::U32,
                dimensions: Dimensions::Fixed(vec![1]),
            }],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError> {
        let Tensor {
            element_type,
            buffer,
            ..
        } = support::get_input_tensor(&inputs, "input")?;

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
                    "The Arg Max proc-block doesn't support {:?} element type",
                    other,
                )))
            },
        };

        let index = match index {
            Some(ix) => ix as u32,
            None => {
                return Err(KernelError::Other(
                    "The input tensor was empty".to_string(),
                ))
            },
        };

        Ok(vec![Tensor::new("max_index", ndarray::array![index])])
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
