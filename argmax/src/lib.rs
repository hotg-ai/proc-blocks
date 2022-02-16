use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Argmax {}

impl Argmax {
    pub const fn new() -> Self { Argmax {} }
}

impl Default for Argmax {
    fn default() -> Self { Argmax::new() }
}

impl Transform<Tensor<f32>> for Argmax {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        let (index, _) = input.elements().iter().enumerate().fold(
            (0, 0.0),
            |max, (ind, &val)| if val > max.1 { (ind, val) } else { max },
        );

        let v: Vec<u32> = vec![index as u32];

        Tensor::new_vector(v)
    }
}

#[cfg(feature = "metadata")]
pub mod metadata {
    wit_bindgen_rust::import!("wit-files/rune/runtime-v1.wit");
    wit_bindgen_rust::export!("wit-files/rune/rune-v1.wit");

    struct RuneV1;

    impl rune_v1::RuneV1 for RuneV1 {
        fn start() {
            use runtime_v1::*;

            let metadata = Metadata::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            );
            metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.add_tag("max");
            metadata.add_tag("numeric");

            let element_types = &[
                ElementType::Uint8,
                ElementType::Int8,
                ElementType::Uint16,
                ElementType::Int16,
                ElementType::Uint32,
                ElementType::Int32,
                ElementType::Float32,
                ElementType::Int64,
                ElementType::Uint64,
                ElementType::Float64,
            ];

            let input = TensorMetadata::new("input");

            for &ty in element_types {
                let hint = example_shape(ty, Dimensions::Dynamic);
                input.add_hint(&hint);
            }
            metadata.add_input(&input);

            let max = TensorMetadata::new("max");
            max.set_description(
                "The index of the element with the highest value",
            );
            metadata.add_output(&max);

            register_node(&metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn test_argmax() {
        let v = Tensor::new_vector(vec![2.3, 12.4, 55.1, 15.4]);
        let mut argmax = Argmax::default();
        let output = argmax.transform(v);
        let should_be: Tensor<u32> = [2].into();
        assert_eq!(output, should_be);
    }
}
