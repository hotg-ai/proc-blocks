use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{
    ndarray::ArrayView1, runtime_v1::*, BufferExt, SliceExt,
};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

const I16_MAX_AS_FLOAT: f32 = i16::MAX as f32;

#[derive(Debug, Clone, PartialEq)]
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Audio Float Conversion", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("audio");
        metadata.add_tag("float");

        let input = TensorMetadata::new("input");
        let hint = supported_shapes(
            &[ElementType::I16],
            DimensionsParam::Fixed(&[1, 0]),
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("output");
        output.set_description(
            "converted values from i16 data type to a floating-point value.",
        );
        let hint = supported_shapes(
            &[ElementType::F32],
            DimensionsParam::Fixed(&[1, 0]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "input",
            ElementType::I16,
            DimensionsParam::Fixed(&[1, 0]),
        );
        ctx.add_output_tensor(
            "output",
            ElementType::F32,
            DimensionsParam::Fixed(&[1, 0]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx =
            KernelContext::for_node(&id).ok_or(KernelError::MissingContext)?;

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

        let tensor: ArrayView1<i16> = match element_type {
            ElementType::I16 => {
                let tensor = buffer.view::<i16>(&dimensions)
                    .map_err(|e| KernelError::InvalidInput(InvalidInput {
                        name: "input".to_string(),
                        reason: BadInputReason::Other(e.to_string()),
                    }))?;

                    tensor.into_dimensionality()
                    .map_err(|_| KernelError::InvalidInput(InvalidInput {
                        name: "input".to_string(),
                        reason: BadInputReason::UnsupportedShape,
                    }))?
            },

            other => {
                return Err(KernelError::Other(format!(
                "The Audio Float Conversion proc-block only accepts I16 tensors, found {:?}",
                other,
                )))
            },
        };

        let output = audio_float_conversion(tensor);

        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type: ElementType::F32,
                dimensions: &dimensions,
                buffer: output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn audio_float_conversion(values: ArrayView1<'_, i16>) -> Vec<f32> {
    values
        .iter()
        .map(|&value| (value as f32 / I16_MAX_AS_FLOAT).clamp(-1.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use hotg_rune_proc_blocks::ndarray::array;

    #[test]
    fn handle_empty() {
        let input = array![0, 0, 0, 0, 0, 0];
        let should_be = vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];

        let got = audio_float_conversion(input.view());

        assert_eq!(got, should_be);
    }

    #[test]
    fn does_it_match() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = array![0, max / 2, min / 2];

        let got = audio_float_conversion(input.view());

        assert_eq!(got, vec![0.0, 0.49998474, -0.50001526]);
    }
    #[test]
    fn clamp_to_bounds() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = array![max, min, min + 1];

        let got = audio_float_conversion(input.view());

        assert_eq!(got, vec![1.0, -1.0, -1.0]);
    }
}
