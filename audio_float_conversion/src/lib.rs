use hotg_rune_proc_blocks::{
    guest::{
        Argument, Dimensions, ElementTypeConstraint, Metadata, ProcBlock,
        RunError, Tensor, TensorConstraint, TensorConstraints, TensorMetadata,
    },
    ndarray::{Array1, ArrayView1},
};

const I16_MAX_AS_FLOAT: f32 = i16::MAX as f32;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: AudioFloatConversion,
}

fn metadata() -> Metadata {
    Metadata::new("Audio Float Conversion", env!("CARGO_PKG_VERSION"))
        .with_description(env!("CARGO_PKG_DESCRIPTION"))
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("audio")
        .with_tag("float")
        .with_input(TensorMetadata::new("input"))
        .with_output(TensorMetadata::new("output").with_description(
            "converted values from i16 data type to a floating-point value.",
        ))
}

#[derive(Debug, Clone, PartialEq)]
struct AudioFloatConversion;

impl ProcBlock for AudioFloatConversion {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint {
                name: "input".to_string(),
                element_type: ElementTypeConstraint::I16,
                dimensions: Dimensions::Fixed(vec![1, 0]),
            }],
            outputs: vec![TensorConstraint {
                name: "output".to_string(),
                element_type: ElementTypeConstraint::F32,
                dimensions: Dimensions::Fixed(vec![1, 0]),
            }],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "input")?;

        let output = audio_float_conversion(tensor.view_1d()?);

        Ok(vec![Tensor::new("output", &output)])
    }
}

impl From<Vec<Argument>> for AudioFloatConversion {
    fn from(_: Vec<Argument>) -> Self { AudioFloatConversion }
}

fn audio_float_conversion(values: ArrayView1<'_, i16>) -> Array1<f32> {
    values.mapv(|value| (value as f32 / I16_MAX_AS_FLOAT).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proc_block_v2::Node as _;
    use hotg_rune_proc_blocks::ndarray::{self, array};

    #[test]
    fn handle_empty() {
        let input = vec![Tensor::new_1d("input", &[0, 0, 0, 0, 0, 0])];
        let should_be = vec![Tensor::new_1d(
            "output",
            &[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0],
        )];
        let node = Node;

        let got = node.run(input).unwrap();

        assert_eq!(got, should_be);
    }

    #[test]
    fn does_it_match() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = array![0, max / 2, min / 2];

        let got = audio_float_conversion(input.view());

        assert_eq!(got, ndarray::array![0.0, 0.49998474, -0.50001526]);
    }
    #[test]
    fn clamp_to_bounds() {
        let max = i16::MAX;
        let min = i16::MIN;

        let input = array![max, min, min + 1];

        let got = audio_float_conversion(input.view());

        assert_eq!(got, ndarray::array![1.0, -1.0, -1.0]);
    }
}
