use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use num_traits::{FromPrimitive, ToPrimitive};

pub fn modulo<T>(modulus: f32, values: &mut [T])
where
    T: ToPrimitive + FromPrimitive,
{
    for item in values {
        let float = item.to_f32().unwrap();
        *item = T::from_f32(float % modulus).unwrap();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, ProcBlock)]
pub struct Modulo {
    modulus: f32,
}

impl Modulo {
    pub fn new() -> Self { Modulo { modulus: 1.0 } }
}

impl Default for Modulo {
    fn default() -> Self { Modulo::new() }
}

impl<'a, T> Transform<Tensor<T>> for Modulo
where
    T: ToPrimitive + FromPrimitive,
{
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<T>) -> Tensor<T> {
        let modulus = self.modulus;

        input.map(|_, item| {
            let float = item.to_f32().unwrap();
            T::from_f32(float % modulus).unwrap()
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

            let metadata = Metadata::new("Modulo", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Apply the modulus operator to each element in the tensor.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));

            let modulus = ArgumentMetadata::new("modulus");
            modulus.set_type_hint(TypeHint::Float);
            modulus.set_default_value("0");
            metadata.add_argument(&modulus);

            let input = TensorMetadata::new("input");
            let supported_types = [
                ElementType::Uint8,
                ElementType::Int8,
                ElementType::Uint16,
                ElementType::Int16,
                ElementType::Uint32,
                ElementType::Int32,
                ElementType::Float32,
                ElementType::Uint64,
                ElementType::Int64,
                ElementType::Float64,
            ];
            let hint = supported_shapes(&supported_types, Dimensions::Dynamic);
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("remainders");
            let hint = supported_shapes(&supported_types, Dimensions::Dynamic);
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
    fn mod_360() {
        let number = 42 + 360;
        let mut m = Modulo::new();
        m.set_modulus("360").unwrap();
        let input = Tensor::single(number);

        let got = m.transform(input);

        assert_eq!(got, Tensor::single(42_i64));
    }
}
