use std::cmp::Ordering;

use crate::{
    hotg_proc_blocks::BufferExt,
    rune_v1::{GraphError, KernelError},
    runtime_v1::*,
};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/rune-v1.wit");

struct RuneV1;

impl rune_v1::RuneV1 for RuneV1 {
    fn start() {
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

    fn graph() -> Result<(), GraphError> {
        let ctx = GraphContext::current().ok_or_else(|| {
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

    fn kernel() -> Result<(), KernelError> {
        let ctx = KernelContext::current().ok_or_else(|| {
            KernelError::Other("Unable to get the kernel context".to_string())
        })?;

        let TensorResult {
            element_type,
            dimensions,
            mut buffer,
        } = ctx
            .get_input_tensor("input")
            .ok_or_else(|| KernelError::MissingInput("input".to_string()))?;

        let floats: &mut [f32] = match element_type {
            ElementType::F32 => buffer.elements_mut::<f32>(),
            other => {
                return Err(KernelError::Other(format!(
                "The Arg Max proc-block only accepts f32 tensors, found {:?}",
                other
            )))
            },
        };

        let index = match arg_max(floats) {
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

fn arg_max(floats: &[f32]) -> Option<usize> {
    let (index, _) = floats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less))?;

    Some(index)
}

/// Support crate provided by hotg.
mod hotg_proc_blocks {
    pub trait BufferExt {
        fn elements_mut<T: ValueType>(&mut self) -> &mut [T];
    }

    impl BufferExt for [u8] {
        fn elements_mut<T: ValueType>(&mut self) -> &mut [T] {
            unsafe { T::from_bytes_mut(self) }
        }
    }

    pub unsafe trait ValueType: Sized {
        unsafe fn from_bytes_mut(bytes: &mut [u8]) -> &mut [Self];
    }

    macro_rules! impl_value_type {
        ($( $type:ty ),* $(,)?) => {
            $(
                unsafe impl ValueType for $type {
                    unsafe fn from_bytes_mut(bytes: &mut [u8]) -> &mut [Self] {
                        let (start, middle, end) = bytes.align_to_mut::<$type>();
                        assert!(start.is_empty());
                        assert!(end.is_empty());
                        middle
                    }
                }
            )*
        };
    }

    impl_value_type!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
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
        let result = arg_max(&[]);

        assert!(result.is_none());
    }
}
