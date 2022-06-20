use hotg_rune_proc_blocks::guest::{
    Argument, Dimensions, ElementType, Metadata, ProcBlock, RunError, Tensor,
    TensorConstraint, TensorConstraints, TensorMetadata,
};

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: PasswordStrength,
}

fn metadata() -> Metadata {
    Metadata::new("Password Strength", env!("CARGO_PKG_VERSION"))
        .with_description("Gauge the strength of your password!")
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("text")
        .with_tag("string")
        .with_input(TensorMetadata::new("password"))
        .with_output(
            TensorMetadata::new("password_strength")
                .with_description("Label for Password strength"),
        )
}

/// A proc block which can convert u8 bytes to utf8
#[derive(Debug, Default, Clone, PartialEq)]
struct PasswordStrength;

impl ProcBlock for PasswordStrength {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "password",
                ElementType::Utf8,
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::new(
                "password_strength",
                ElementType::U32,
                Dimensions::Dynamic,
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let password = Tensor::get_named(&inputs, "password")?.string_view()?;

        let strength = password.mapv(password_strength);

        Ok(vec![Tensor::new("password_strength", &strength)])
    }
}

impl From<Vec<Argument>> for PasswordStrength {
    fn from(_: Vec<Argument>) -> Self { PasswordStrength::default() }
}

fn password_strength(password: &str) -> u32 {
    match password.len() {
        0..=6 => 2,
        7..=10 => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {

    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn test_for_utf8_decoding() {
        let passwords = ndarray::array![
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tv",
        ];
        let input = vec![Tensor::from_strings("password", &passwords)];
        let should_be = vec![Tensor::new_1d(
            "password_strength",
            &[
                1_u32, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 1, 2, 0, 2, 2, 2, 2,
            ],
        )];

        let output = PasswordStrength::default().run(input).unwrap();

        assert_eq!(output, should_be);
    }
}
