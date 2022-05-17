use crate::proc_block_v1::*;
use hotg_rune_proc_blocks::{runtime_v1::*, BufferExt, SliceExt};
use num_traits::{Bounded, ToPrimitive};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

/// A normalization routine which takes some tensor of integers and fits their
/// values to the range `[0, 1]` as `f32`'s.

#[derive(Debug, Clone, PartialEq)]
struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata =
            Metadata::new("Image Normalization", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "Normalize the pixels in an image to the range `[0, 1]`",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("image");
        metadata.add_tag("normalize");

        let input = TensorMetadata::new("image");
        input.set_description("An image with the dimensions `[1, width, height, channels]`.\n\nRGB images typically have 3 channels and grayscale images have 1.");
        let hint = supported_shapes(
            &[
                ElementType::U8,
                ElementType::I8,
                ElementType::U16,
                ElementType::I16,
                ElementType::U32,
                ElementType::I32,
            ],
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("normalized_image");
        output.set_description(
            "The image's pixels, normalized to the range `[0, 1]`.",
        );
        let hint = supported_shapes(
            &[ElementType::F32],
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

        let element_type = match ctx.get_argument("element_type").as_deref() {
            Some("u8") => ElementType::U8,
            Some("i8") => ElementType::I8,
            Some("u16") => ElementType::U16,
            Some("i16") => ElementType::I16,
            Some("u32") => ElementType::U32,
            Some("i32") => ElementType::I32,
            Some("f32") => ElementType::F32,
            Some("u64") => ElementType::U64,
            Some("i64") => ElementType::I64,
            Some("f64") => ElementType::F64,
            Some(_) => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::InvalidValue(
                        "Unsupported element type".to_string(),
                    ),
                }));
            },
            None => {
                return Err(GraphError::InvalidArgument(InvalidArgument {
                    name: "element_type".to_string(),
                    reason: BadArgumentReason::NotFound,
                }))
            },
        };

        ctx.add_input_tensor(
            "image",
            element_type,
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );
        ctx.add_output_tensor(
            "normalized_image",
            ElementType::F32,
            DimensionsParam::Fixed(&[1, 0, 0, 0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_input_tensor("image").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "image".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        check_input_dimensions(&dimensions);

        let output = match element_type {
            ElementType::U8 => buffer
            .view::<u8>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),

            ElementType::I8 => buffer
            .view::<i8>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),
            ElementType::U16 => buffer
            .view::<u16>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),
            ElementType::I16 => buffer
            .view::<i16>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),
            ElementType::U32 => buffer
            .view::<u32>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),
            ElementType::I32 => buffer
            .view::<i32>(&dimensions)
            .map_err( |e | KernelError::InvalidInput(InvalidInput {name: "image".to_string(), reason: BadInputReason::Other(e.to_string()),}))?.map( |&t| normalize(t)),
            other => {
                return Err(KernelError::Other(format!(
                "The Audio Float Conversion proc-block only accepts I16 tensors, found {:?}",
                other,
                )))
            }
        };
        let output: Vec<f32> = output.iter().map(|&v| v as f32).collect();
        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type: ElementType::F32,
                dimensions: &dimensions,
                buffer: output.as_bytes(),
            },
        );

        Ok(())
    }
}

fn check_input_dimensions(dimensions: &[u32]) {
    match *dimensions {
        [_, _, _, 3] => {},
        [_, _, _, channels] => panic!(
            "The number of channels should be either 1 or 3, found {}",
            channels
        ),
        _ => panic!("The image normalization proc block only supports outputs of the form [frames, rows, columns, channels], found {:?}", dimensions),
    }
}

fn normalize<T>(value: T) -> f32
where
    T: Bounded + ToPrimitive,
{
    let min = T::min_value().to_f32().unwrap();
    let max = T::max_value().to_f32().unwrap();
    let value = value.to_f32().unwrap();

    (value - min) / (max - min)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizing_with_default_distribution_is_noop() {
        let input: u8 = 5;
        let should_be: f32 = 5.0 / 255.0;
        let got = normalize(input);
        assert_eq!(got, should_be);
    }
}
