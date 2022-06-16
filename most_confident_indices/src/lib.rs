use hotg_rune_proc_blocks::guest::{
    Argument, ArgumentMetadata, ArgumentType, CreateError, Dimensions,
    ElementType, InvalidInput, Metadata, ProcBlock, RunError, Tensor,
    TensorConstraint, TensorConstraints, TensorMetadata,
};

use std::{cmp::Ordering, convert::TryInto};

use hotg_rune_proc_blocks::ndarray::ArrayView1;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: MostConfidentIndices,
}

fn metadata() -> Metadata {
    Metadata::new("Most Confident Indices", env!("CARGO_PKG_VERSION"))
    .with_description(
        "Given some confidence values, create a tensor containing the indices of the top N highest confidences.",
    )
    .with_repository(env!("CARGO_PKG_REPOSITORY"))
    .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
    .with_tag("classify")
    .with_tag("confidence")
    .with_argument(ArgumentMetadata::new("count")
    .with_default_value("1")
    .with_description("The number of indices to return")
    .with_hint(ArgumentType::Float))
    .with_input(TensorMetadata::new("confidences").with_description("A 1D tensor of numeric confidence values."))
    .with_output(
        TensorMetadata::new("indices")
            .with_description("The indices, in order of descending confidence."),
    )
}

struct MostConfidentIndices {
    count: u32,
}

impl ProcBlock for MostConfidentIndices {
    fn tensor_constraints(&self) -> TensorConstraints {
        let count = 1 as u32; // todo: replace with actual arguements passed by user

        TensorConstraints {
            inputs: vec![TensorConstraint::numeric(
                "confidences",
                Dimensions::Dynamic,
            )],
            outputs: vec![TensorConstraint::numeric("indices", vec![count])],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "confidences")?;

        let count = 1 as usize; // todo: replace with actual arguements passed by user

        let indices = match tensor.element_type {
            ElementType::U8 => most_confident_indices(tensor.view_1d::<u8>()?, count)?,
            ElementType::I8 => most_confident_indices(tensor.view_1d::<i8>()?, count)?,
            ElementType::U16 => most_confident_indices(tensor.view_1d::<u16>()?, count)?,
            ElementType::I16 =>most_confident_indices(tensor.view_1d::<i16>()?, count)?,
            ElementType::U32 => most_confident_indices(tensor.view_1d::<u32>()?, count)?,
            ElementType::I32 => most_confident_indices(tensor.view_1d::<i32>()?, count)?,
            ElementType::F32 => most_confident_indices(tensor.view_1d::<f32>()?, count)?,
            ElementType::U64 => most_confident_indices(tensor.view_1d::<u64>()?, count)?,
            ElementType::I64 => most_confident_indices(tensor.view_1d::<i64>()?, count)?,
            ElementType::F64 => most_confident_indices(tensor.view_1d::<f64>()?, count)?,
            _ => {
                return Err(InvalidInput::incompatible_element_type(
                    "confidences",
                )
                .into());
            },
        };

        Ok(vec![Tensor::new_1d("indices", &indices.to_vec())])
    }
}

// fn preprocess_buffer<
//     'buf,
//     T: hotg_rune_proc_blocks::guest::PrimitiveTensorElement,
// >(
//     buffer: &Tensor,
// ) -> Result<ArrayView1<&T>, RunError> {

//     buffer
//         .view::<T>().unwrap().and_then(|t| Ok(t.into_dimensionality()))
//         .map_err(|e| RunError::other(format!("Invalid input: {}", e))).unwrap()
// }


fn most_confident_indices<T>(
    tensor: ArrayView1<T>,
    count: usize,
) -> Result<Vec<u32>, RunError>
where
    T: PartialOrd + Copy,
{
    if count > tensor.len() {
        return Err(RunError::other(format!(
            "Requesting {} indices from a tensor with only {} elements",
            count,
            tensor.len()
        )));
    }

    let mut indices_and_confidence: Vec<_> =
        tensor.iter().copied().enumerate().collect();

    indices_and_confidence
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));

    Ok(indices_and_confidence
        .into_iter()
        .map(|(index, _confidence)| index.try_into().unwrap())
        .take(count)
        .collect())
}

impl TryFrom<Vec<Argument>> for MostConfidentIndices {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let count =
            hotg_rune_proc_blocks::guest::parse::optional_arg(&args, "count")?
                .unwrap_or(1);

        Ok(MostConfidentIndices { count })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use hotg_rune_proc_blocks::ndarray;

//     #[test]
//     fn only_works_with_1d() {
//         let buffer = [1, 2, 3, 4, 5, 6];

//         let error = preprocess_buffer::<u8>(&buffer).unwrap_err();

//         assert!(matches!(error, RunError::InvalidInput(_)));
//     }

//     #[test]
//     fn tensors_equivalent_to_1d_are_okay_too() {
//         let buffer = [1, 2, 3, 4, 5, 6];

//         let error = preprocess_buffer::<u8>(&buffer).unwrap_err();

//         assert!(matches!(error, RunError::InvalidInput(_)));
//     }

//     #[test]
//     fn count_must_be_less_than_input_size() {
//         let input = ndarray::arr1(&[1_u32, 2, 3]);

//         let error = most_confident_indices(input.view(), 42).unwrap_err();

//         assert!(matches!(error, RunError::InvalidArgument(_)));
//     }

//     #[test]
//     fn get_top_3_values() {
//         let elements = ndarray::arr1(&[0.0, 0.5, 10.0, 3.5, -200.0]);

//         let got = most_confident_indices(elements.view(), 3).unwrap();

//         assert_eq!(got, &[2, 3, 1]);
//     }
// }
