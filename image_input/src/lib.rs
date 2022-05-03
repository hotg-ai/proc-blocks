use std::{
    error::Error,
    fmt::{self, Display, Formatter},
    str::FromStr,
};

use crate::{
    proc_block_v1::{
        BadArgumentReason, BadInputReason, GraphError, InvalidArgument,
        InvalidInput, KernelError,
    },
    runtime_v1::*,
};
use hotg_rune_proc_blocks::{prelude::*, runtime_v1};

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("Image Input", env!("CARGO_PKG_VERSION"));
        metadata.set_description(env!("CARGO_PKG_DESCRIPTION"));
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("input");
        metadata.add_tag("image");

        let width = ArgumentMetadata::new("width");
        width.set_description("The image width in pixels.");
        let hint = runtime_v1::non_negative_number();
        width.add_hint(&hint);
        metadata.add_argument(&width);

        let height = ArgumentMetadata::new("height");
        height.set_description("The image height in pixels.");
        let hint = runtime_v1::non_negative_number();
        height.add_hint(&hint);
        metadata.add_argument(&height);

        let pixel_format = ArgumentMetadata::new("pixel_format");
        pixel_format.set_description("The pixel format.");
        let hint = runtime_v1::non_negative_number();
        pixel_format.add_hint(&hint);
        metadata.add_argument(&pixel_format);

        let output = TensorMetadata::new("image");
        let hint = supported_shapes(
            &[ElementType::U8, ElementType::F32],
            DimensionsParam::Fixed(&[0, 0, 0, 0]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(id: String) -> Result<(), GraphError> {
        let ctx =
            GraphContext::for_node(&id).ok_or(GraphError::MissingContext)?;

        let width: u32 = ctx.parse_argument("width")?;
        let height: u32 = ctx.parse_argument("height")?;
        let pixel_format: PixelFormat = ctx.parse_argument("pixel_format")?;

        ctx.add_output_tensor(
            "output",
            pixel_format.element_type(),
            DimensionsParam::Fixed(&[
                1,
                width,
                height,
                pixel_format.channels(),
            ]),
        );

        Ok(())
    }

    fn kernel(id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&id).ok_or_else(|| {
            KernelError::Other("Unable to get the kernel context".to_string())
        })?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_global_input(&id).ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: id,
                reason: BadInputReason::NotFound,
            })
        })?;

        // TODO: use the width, height, and pixel format to resize the image for
        // now, we're just going to copy it out as-is and hope for the best.
        let _width: u32 = ctx.parse_argument("width")?;
        let _height: u32 = ctx.parse_argument("height")?;
        let _pixel_format: PixelFormat = ctx.parse_argument("pixel_format")?;

        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type,
                dimensions: &dimensions,
                buffer: &buffer,
            },
        );

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum PixelFormat {
    RGB8,
}

impl PixelFormat {
    fn channels(self) -> u32 {
        match self {
            PixelFormat::RGB8 => 3,
        }
    }

    fn element_type(self) -> ElementType {
        match self {
            PixelFormat::RGB8 => ElementType::U8,
        }
    }
}

impl FromStr for PixelFormat {
    type Err = UnknownPixelFormat;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rgb" | "rgb8" => Ok(PixelFormat::RGB8),
            _ => Err(UnknownPixelFormat),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct UnknownPixelFormat;

impl Display for UnknownPixelFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        "Unknown pixel format".fmt(f)
    }
}

impl Error for UnknownPixelFormat {}

impl ContextErrorExt for GraphError {
    type InvalidArgument = InvalidArgument;

    fn invalid_argument(inner: InvalidArgument) -> Self {
        GraphError::InvalidArgument(inner)
    }
}

impl ContextErrorExt for KernelError {
    type InvalidArgument = InvalidArgument;

    fn invalid_argument(inner: InvalidArgument) -> Self {
        KernelError::InvalidArgument(inner)
    }
}

impl InvalidArgumentExt for InvalidArgument {
    fn other(name: &str, msg: impl std::fmt::Display) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::Other(msg.to_string()),
        }
    }

    fn invalid_value(name: &str, error: impl std::fmt::Display) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::InvalidValue(error.to_string()),
        }
    }

    fn not_found(name: &str) -> Self {
        InvalidArgument {
            name: name.to_string(),
            reason: BadArgumentReason::NotFound,
        }
    }
}
