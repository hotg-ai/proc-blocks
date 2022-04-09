use std::{cmp::Ordering, convert::TryInto};

use crate::{
    proc_block_v1::{GraphError, KernelError},
    runtime_v1::*,
};

wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");
wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Most Confident Indices", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
                "Given some confidence values, create a tensor containing the indices of the top N highest confidences.",
            );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("classify");

        let count = ArgumentMetadata::new("count");
        count.set_description("The number of indices to return.");
        count.set_default_value("1");
        let hint =
            runtime_v1::supported_argument_type(ArgumentType::UnsignedInteger);
        count.add_hint(&hint);
        metadata.add_argument(&count);

        let input = TensorMetadata::new("confidences");
        input.set_description("A 1D tensor of numeric confidence values.");
        let hint = supported_shapes(
            &[
                ElementType::U8,
                ElementType::I8,
                ElementType::U16,
                ElementType::I16,
                ElementType::U32,
                ElementType::I32,
                ElementType::F32,
                ElementType::U64,
                ElementType::I64,
                ElementType::F64,
            ],
            Dimensions::Dynamic,
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("indices");
        output
            .set_description("The indices, in order of descending confidence.");
        let hint =
            supported_shapes(&[ElementType::U32], Dimensions::Fixed(&[0]));
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        todo!();
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        todo!();
    }
}

fn most_confident_indices<T>(elements: &[T], count: usize) -> Vec<u32>
where
    T: PartialOrd + Copy,
{
    assert!(count <= elements.len());

    let mut indices_and_confidence: Vec<_> =
        elements.iter().copied().enumerate().collect();

    indices_and_confidence
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));

    indices_and_confidence
        .into_iter()
        .map(|(index, _confidence)| index.try_into().unwrap())
        .take(count)
        .collect()
}

fn dimensions_are_valid(dimensions: &[u32]) -> bool {
    match dimensions {
        [1, length] | [length] if length > 0 => true,
        _ => false,
    }
}

fn get_count(get )

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn only_works_with_1d() {
        let dimensions = [1, 2, 3];

        assert!(!dimensions_are_valid(&dimensions));
    }

    #[test]
    fn tensors_equivalent_to_1d_are_okay_too() {
        let dimensions = [1, 5, 1, 1, 1];

        assert!(!dimensions_are_valid(&dimensions));
    }

    #[test]
    #[should_panic]
    fn count_must_be_less_than_input_size() {
        let mut proc_block = MostConfidentIndices::new(42);
        let input = Tensor::new_vector(vec![0, 0, 1, 2]);

        let _ = proc_block.transform(input);
    }

    #[test]
    fn get_top_3_values() {
        let elements = [0.0, 0.5, 10.0, 3.5, -200.0];

        let got = most_confident_indices(&elements, 3);

        assert_eq!(got, &[2, 3, 1]);
    }
}
