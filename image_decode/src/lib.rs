use hotg_rune_proc_blocks::{
    guest::{
        parse, Argument, ArgumentHint, ArgumentMetadata, CreateError,
        ElementType, ElementTypeConstraint, InvalidInput, Metadata,
        PrimitiveTensorElement, ProcBlock, RunError, Tensor, TensorConstraint,
        TensorConstraints, TensorMetadata,
    },
    ndarray::Array,
};
use image::{
    flat::SampleLayout, imageops::FilterType, FlatSamples, ImageBuffer, Pixel,
};
use strum::VariantNames;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Image,
}

fn metadata() -> Metadata {
    Metadata::new("Image Decode", env!("CARGO_PKG_VERSION"))
        .with_description(env!("CARGO_PKG_DESCRIPTION"))
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("input")
        .with_tag("image")
        .with_argument(
            ArgumentMetadata::new("width")
                .with_description("The image width in pixels.")
                .with_hint(ArgumentHint::NonNegativeNumber),
        )
        .with_argument(
            ArgumentMetadata::new("height")
                .with_description("The image height in pixels.")
                .with_hint(ArgumentHint::NonNegativeNumber),
        )
        .with_argument(
            ArgumentMetadata::new("pixel_format")
                .with_description(
                    "The pixel format to use for the loaded image.",
                )
                .with_default_value(PixelFormat::RGB8.to_string())
                .with_hint(
                    ArgumentHint::OneOf(
                        PixelFormat::VARIANTS
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
                    ),
                )
                .with_hint(ArgumentHint::NonNegativeNumber),
        )
        .with_input(
            TensorMetadata::new("file")
                .with_description("A file containing the image"),
        )
        .with_output(TensorMetadata::new("image"))
}

#[derive(Debug, Clone, PartialEq)]
struct Image {
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
}

impl ProcBlock for Image {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "file",
                ElementTypeConstraint::U8,
                vec![0],
            )],
            outputs: vec![TensorConstraint::new(
                "image",
                self.pixel_format.element_type(),
                self.pixel_format
                    .dimensions(self.width, self.height)
                    .to_vec(),
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let tensor = Tensor::get_named(&inputs, "file")?;
        let view = tensor.view_1d::<u8>()?;
        let bytes = view.as_slice().ok_or_else(|| {
            RunError::other(
                "Unable to view the file tensor as a contiguous slice",
            )
        })?;

        let img = image::load_from_memory(bytes)
            .map_err(|e| InvalidInput::other("file", e))?;

        let resized = img.resize_exact(
            self.width as u32,
            self.height as u32,
            FilterType::Nearest,
        );

        let formatted = match self.pixel_format {
            PixelFormat::RGB8 => to_tensor(resized.into_rgb8()),
            PixelFormat::RGBA8 => to_tensor(resized.into_rgba8()),
        };

        Ok(vec![formatted])
    }
}

fn to_tensor<P>(img: ImageBuffer<P, Vec<P::Subpixel>>) -> Tensor
where
    P: Pixel,
    P::Subpixel: PrimitiveTensorElement,
{
    let FlatSamples {
        samples,
        layout:
            SampleLayout {
                channels,
                width,
                height,
                ..
            },
        ..
    } = img.into_flat_samples();

    let array = Array::from_shape_vec(
        (width as usize, height as usize, channels as usize),
        samples,
    )
    .expect("Image dimensions should always be well-formed");

    Tensor::new("image", &array)
}

impl TryFrom<Vec<Argument>> for Image {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let pixel_format = parse::optional_arg(&args, "pixel_format")?
            .unwrap_or(PixelFormat::RGB8);
        let width = parse::required_arg(&args, "width")?;
        let height = parse::required_arg(&args, "height")?;

        Ok(Image {
            pixel_format,
            height,
            width,
        })
    }
}

#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::EnumVariantNames,
    strum::Display,
)]
enum PixelFormat {
    #[strum(serialize = "rgb8")]
    RGB8,
    #[strum(serialize = "rgba8")]
    RGBA8,
}

impl PixelFormat {
    fn dimensions(self, width: usize, height: usize) -> [u32; 3] {
        match self {
            PixelFormat::RGBA8 | PixelFormat::RGB8 => {
                [width as u32, height as u32, self.channels()]
            },
        }
    }

    fn channels(self) -> u32 {
        match self {
            PixelFormat::RGB8 => 3,
            PixelFormat::RGBA8 => 4,
        }
    }

    fn element_type(self) -> ElementType {
        match self {
            PixelFormat::RGB8 | PixelFormat::RGBA8 => ElementType::U8,
        }
    }
}

#[cfg(test)]
mod tests {
    use hotg_rune_proc_blocks::ndarray;

    use super::*;

    #[test]
    fn load_a_known_file() {
        let bytes = include_bytes!("image.png");
        // [black, red]
        // [green, blue]
        let tensor = Tensor::new_1d("file", bytes);
        let proc_block = Image {
            height: 2,
            width: 2,
            pixel_format: PixelFormat::RGB8,
        };

        let got = proc_block.run(vec![tensor]).unwrap();

        assert_eq!(got.len(), 1);
        let image = Tensor::get_named(&got, "image")
            .unwrap()
            .view_3d::<u8>()
            .unwrap();

        let should_be = ndarray::array![
            [[255_u8, 0, 0], [0, 0, 0]],
            [[0, 255, 0], [0, 0, 255],]
        ];
        assert_eq!(image, should_be);
    }
}
