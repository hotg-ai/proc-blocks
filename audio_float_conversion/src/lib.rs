use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

// TODO: Add Generics

#[derive(Debug, Clone, PartialEq, ProcBlock)]
#[transform(inputs = [i16; _], outputs = [f32; _])]
pub struct AudioFloatConversion {
    i16_max_as_float: f32,
}

const I16_MAX_AS_FLOAT: f32 = i16::MAX as f32;

impl AudioFloatConversion {
    pub const fn new() -> Self {
        AudioFloatConversion {
            i16_max_as_float: I16_MAX_AS_FLOAT,
        }
    }

    fn check_input_dimensions(&self, dimensions: &[usize]) {
        assert_eq!(
            (!(dimensions.len() == 2 && dimensions[0] == 1)
                || !(dimensions.len() == 1)),
            true,
            "This proc block only supports 1D outputs (requested output: {:?})",
            dimensions
        );
    }
}

impl Default for AudioFloatConversion {
    fn default() -> Self { AudioFloatConversion::new() }
}

impl Transform<Tensor<i16>> for AudioFloatConversion {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<i16>) -> Self::Output {
        self.check_input_dimensions(input.dimensions());
        input.map(|_dims, &value| {
            (value as f32 / I16_MAX_AS_FLOAT).clamp(-1.0, 1.0)
        })
    }
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!(
        "$CARGO_MANIFEST_DIR/../wit-files/rune/runtime-v1.wit"
    );
    wit_bindgen_rust::export!(
        "$CARGO_MANIFEST_DIR/../wit-files/rune/rune-v1.wit"
    );

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata = Metadata::new(
                "Audio to Float Conversion",
                env!("CARGO_PKG_VERSION"),
            );
            metadata.set_description(
                "Convert a tensor of `i16` samples to a normalized tensor of `f32`.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("audio");

            let input = TensorMetadata::new("input");
            input.set_description("A 1D tensor of `i16` samples.");
            let hint = supported_shapes(
                &[ElementType::Int16],
                Dimensions::Fixed(&[0]),
            );
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("normalized");
            output.set_description(
                "The samples as floats normalized to the range `[0, 1]`.",
            );
            let hint = supported_shapes(
                &[ElementType::Float32],
                Dimensions::Fixed(&[0]),
            );
            output.add_hint(&hint);
            metadata.add_output(&output);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_empty() {
        let mut pb = AudioFloatConversion::new();
        let input = Tensor::new_vector(vec![0; 15]);

        let got = pb.transform(input);

        assert_eq!(got.dimensions(), &[15]);
    }

    #[test]
    fn does_it_match() {
        let max = i16::MAX;
        let min = i16::MIN;

        let mut pb = AudioFloatConversion::new();
        let input = Tensor::new_vector(vec![0, max / 2, min / 2]);

        let got = pb.transform(input);

        assert_eq!(got.elements()[0..3], [0.0, 0.49998474, -0.50001526]);
    }
    #[test]
    fn does_clutch_work() {
        let max = i16::MAX;
        let min = i16::MIN;

        let mut pb = AudioFloatConversion::new();
        let input = Tensor::new_vector(vec![max, min, min + 1]);

        let got = pb.transform(input);

        assert_eq!(got.elements()[0..3], [1.0, -1.0, -1.0]);
    }
}
