use crate::proc_block_v1::{
    BadInputReason, GraphError, InvalidInput, KernelError,
};
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};

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
        let ctx = GraphContext::for_node(&id).ok_or_else(|| {
            GraphError::Other("Unable to get the graph context".to_string())
        })?;

        ctx.add_input_tensor(
            "input",
            ElementType::I16,
            DimensionsParam::Fixed(&[1, 0]),
        );
        ctx.add_output_tensor(
            "max_index",
            ElementType::F32,
            DimensionsParam::Fixed(&[1, 0]),
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

        check_input_dimensions(&dimensions);

        let output = match element_type {
            ElementType::I16 => {
                audio_float_conversion(buffer.elements::<i16>())
            },

            other => {
                return Err(KernelError::Other(format!(
                "The Audio Float Conversion proc-block only accepts I16 tensors, found {:?}",
                other,
                )))
            },
        };

        let output = match output {
            Some(ix) => ix,
            None => {
                return Err(KernelError::Other(
                    "The input tensor was empty".to_string(),
                ))
            },
        };

        let resulting_tensor = output.as_bytes();

        ctx.set_output_tensor(
            "max_index",
            TensorParam {
                element_type: ElementType::F32,
                dimensions: &dimensions,
                buffer: &resulting_tensor,
            },
        );

        Ok(())
    }
}

fn check_input_dimensions(dimensions: &[u32]) {
    assert_eq!(
        (!(dimensions.len() == 2 && dimensions[0] == 1)
            || !(dimensions.len() == 1)),
        true,
        "This proc block only supports 1D outputs (requested output: {:?})",
        dimensions
    );
}
fn audio_float_conversion(values: &[i16]) -> Option<Vec<f32>> {
    let mut output = Vec::new();
    for i in 0..values.len() {
        output.push((values[i] as f32 / I16_MAX_AS_FLOAT).clamp(-1.0, 1.0));
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn handle_empty() {
        let input = [0; 15];
        let got = audio_float_conversion(&input).unwrap();
        let dim = got.len();
        assert_eq!(dim, 15);
    }

    #[test]
    fn does_it_match() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = [0, max / 2, min / 2];

        let got = audio_float_conversion(&input).unwrap();

        assert_eq!(got, vec![0.0, 0.49998474, -0.50001526]);
    }
    #[test]
    fn does_clutch_work() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = [max, min, min + 1];

        let got = audio_float_conversion(&input).unwrap();

        assert_eq!(got, vec![1.0, -1.0, -1.0]);
    }
}
