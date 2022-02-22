use hotg_rune_proc_blocks::{Tensor, Transform};
use std::{
    fmt::Debug,
    ops::{Div, Sub},
};

pub fn normalize<T>(input: &mut [T])
where
    T: PartialOrd + Div<Output = T> + Sub<Output = T> + Copy,
{
    if let Some((min, max)) = min_max(input.iter()) {
        if min != max {
            let range = max - min;

            for item in input {
                *item = (*item - min) / range;
            }
        }
    }
}

/// Normalize the input to the range `[0, 1]`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, hotg_rune_proc_blocks::ProcBlock,
)]
#[non_exhaustive]
#[transform(inputs = [f32; 1], outputs = [f32; 1])]
#[transform(inputs = [f32; 2], outputs = [f32; 2])]
#[transform(inputs = [f32; 3], outputs = [f32; 3])]
pub struct Normalize {}

impl<T> Transform<Tensor<T>> for Normalize
where
    T: PartialOrd + Div<Output = T> + Sub<Output = T> + Copy,
{
    type Output = Tensor<T>;

    fn transform(&mut self, mut input: Tensor<T>) -> Tensor<T> {
        normalize(input.make_elements_mut());
        input
    }
}

impl<T, const N: usize> Transform<[T; N]> for Normalize
where
    T: PartialOrd + Div<Output = T> + Sub<Output = T> + Copy,
{
    type Output = [T; N];

    fn transform(&mut self, mut input: [T; N]) -> [T; N] {
        normalize(&mut input);
        input
    }
}

fn min_max<'a, I, T>(items: I) -> Option<(T, T)>
where
    I: IntoIterator<Item = &'a T> + 'a,
    T: PartialOrd + Copy + 'a,
{
    items.into_iter().fold(None, |bounds, &item| match bounds {
        Some((min, max)) => {
            let min = if item < min { item } else { min };
            let max = if max < item { item } else { max };
            Some((min, max))
        },
        None => Some((item, item)),
    })
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

            let metadata =
                Metadata::new("Normalize", env!("CARGO_PKG_VERSION"));
            metadata.set_description(
                "Normalize a tensor's elements to the range, `[0, 1]`.",
            );
            metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
            metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
            metadata.add_tag("normalize");

            let input = TensorMetadata::new("input");
            let supported_types = [ElementType::Float32, ElementType::Float64];
            let hint = supported_shapes(&supported_types, Dimensions::Dynamic);
            input.add_hint(&hint);
            metadata.add_input(&input);

            let output = TensorMetadata::new("normalized");
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
    fn it_works() {
        let input = Tensor::from([0.0, 1.0, 2.0]);
        let mut pb = Normalize::default();

        let output = pb.transform(input);

        assert_eq!(output, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn it_accepts_vectors() {
        let input = [0.0, 1.0, 2.0];
        let mut pb = Normalize::default();

        let _ = pb.transform(input);
    }

    #[test]
    fn handle_empty() {
        let input: [f32; 384] = [0.0; 384];
        let mut pb = Normalize::default();

        let output = pb.transform(input);

        assert_eq!(output, input);
        assert_eq!(output.len(), 384);
    }
}
