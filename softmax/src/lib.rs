use crate::{
    proc_block_v1::{BadInputReason, GraphError, InvalidInput, KernelError},
    runtime_v1::{
        register_node, supported_shapes, Dimensions, ElementType, GraphContext,
        KernelContext, Metadata, TensorMetadata, TensorParam, TensorResult,
    },
};

use hotg_rune_proc_blocks::{ndarray::ArrayViewMut1, BufferExt, ValueType};
use num_traits::Float;

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

fn softmax<T>(mut input: ArrayViewMut1<'_, T>)
where
    T: Float + num_traits::FromPrimitive,
{
    input.mapv_inplace(|x| x.exp());

    let sum = input.sum();
    if !sum.is_zero() {
        input.mapv_inplace(|x| x / sum);
    }
}

fn preprocess_buffer<'buf, T>(
    buffer: &'buf mut [u8],
    dimensions: &[u32],
) -> Result<ArrayViewMut1<'buf, T>, KernelError>
where
    T: ValueType,
{
    buffer
        .view_mut::<T>(dimensions)
        .and_then(|t| t.into_dimensionality())
        .map_err(|e| {
            KernelError::InvalidInput(InvalidInput {
                name: "confidences".to_string(),
                reason: BadInputReason::InvalidValue(e.to_string()),
            })
        })
}

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Softmax", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("softmax");
        metadata.add_tag("image");
        metadata.add_tag("nlp");
        metadata.add_tag("numeric");
        metadata.add_tag("classification");

        let input = TensorMetadata::new("input");
        let hint = supported_shapes(
            &[ElementType::F32, ElementType::F64],
            Dimensions::Fixed(&[0]),
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let soft_max = TensorMetadata::new("soft_max");
        soft_max
            .set_description("Vector normalised into probability distribution");
        let hint = supported_shapes(
            &[ElementType::F32, ElementType::F64],
            Dimensions::Fixed(&[0]),
        );
        soft_max.add_hint(&hint);
        metadata.add_output(&soft_max);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor("input", ElementType::F32, Dimensions::Dynamic);

        ctx.add_output_tensor(
            "soft_max",
            ElementType::F32,
            Dimensions::Dynamic,
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx =
            KernelContext::for_node(&id).ok_or(KernelError::MissingContext)?;
        let TensorResult {
            element_type,
            dimensions,
            mut buffer,
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "input".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        match element_type {
            ElementType::F32 => preprocess_buffer::<f32>(&mut buffer, &dimensions).map(softmax)?,
            ElementType::F64 => preprocess_buffer::<f64>(&mut buffer, &dimensions).map(softmax)?,
            other => {
                return Err(KernelError::Other(format!(
                "The softmax proc-block only accepts f32 or f64 tensors, found {:?}",
                other,
                )))
            },
        };

        ctx.set_output_tensor(
            "soft_max",
            TensorParam {
                element_type,
                dimensions: &dimensions,
                buffer: &buffer,
            },
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hotg_rune_proc_blocks::ndarray;

    #[test]
    fn test_softmax_unfiorm() {
        let mut input = ndarray::arr1(&[1.0, 1.0, 1.0, 1.0]);
        let softmax_correct = ndarray::arr1(&[0.25, 0.25, 0.25, 0.25]);

        softmax(input.view_mut());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn test_softmax_single() {
        let mut input = ndarray::arr1(&[1.0, 0.0]);
        let softmax_correct =
            ndarray::arr1(&[0.7310585786300049, 0.26894142136999510]);
        softmax(input.view_mut());

        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn test_softmax() {
        let mut input = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let softmax_correct = ndarray::arr1(&[
            0.09003057317038046,
            0.24472847105479767,
            0.6652409557748219,
        ]);

        softmax(input.view_mut());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn test_softmax_zeros() {
        let mut input = ndarray::arr1(&[0.0, 0.0]);
        let softmax_correct = ndarray::arr1(&[0.5, 0.5]);

        softmax(input.view_mut());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn test_softmax_zero() {
        let mut input = ndarray::arr1(&[0.0]);
        let softmax_correct = ndarray::arr1(&[1.0]);

        softmax(input.view_mut());
        assert_eq!(input, softmax_correct);
    }

    #[test]
    fn test_softmax_empty() {
        let empty: &[f32] = &[];
        let mut input = ndarray::Array::from_vec(empty.to_vec());
        let softmax_correct = ndarray::Array::from_vec(empty.to_vec());

        softmax(input.view_mut());
        assert_eq!(input, softmax_correct);
    }
}
