use crate::proc_block_v2::*;
use hotg_rune_proc_blocks::{ndarray::ArrayView1, SliceExt};
use wit_bindgen_rust::Handle;

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v2.wit");

hotg_rune_proc_blocks::generate_support!(crate::proc_block_v2);

const I16_MAX_AS_FLOAT: f32 = i16::MAX as f32;

#[derive(Debug, Clone, PartialEq)]
struct ProcBlockV2;

impl proc_block_v2::ProcBlockV2 for ProcBlockV2 {
    fn metadata() -> Metadata {
        Metadata {
            name: "Audio Float Conversion".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        description: Some(env!("CARGO_PKG_DESCRIPTION").to_string()),
        repository: Some(env!("CARGO_PKG_REPOSITORY").to_string()),
        homepage: Some(env!("CARGO_PKG_HOMEPAGE").to_string()),
        tags: vec![
            "audio".to_string(),
            "float".to_string(),

        ],
        arguments: Vec::new(),
        inputs: vec![
            TensorMetadata {
                name: "input".to_string(),
                description: None,
                hints: Vec::new(),
            }
        ],
        outputs: vec![
TensorMetadata {
                name: "output".to_string(),
                description:             Some("converted values from i16 data type to a floating-point value.".to_string()),
                hints: Vec::new(),
            }
        ],
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

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, KernelError> {
        let tensor = support::get_input_tensor(&inputs, "input")?;

        let output = audio_float_conversion(tensor.view_1d()?);

        Ok(vec![Tensor {
            name: "output".to_string(),
            element_type: ElementType::F32,
            dimensions: tensor.dimensions.clone(),
            buffer: output.as_bytes().to_vec(),
        }])
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
