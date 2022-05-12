use std::fmt::Display;

use crate::proc_block_v1::*;

use hotg_rune_proc_blocks::{
    runtime_v1::{self, *},
    BufferExt, SliceExt,
};

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

#[macro_use]
extern crate alloc;
use alloc::vec::Vec;
use nalgebra::DMatrix;
use sonogram::SpecOptionsBuilder;

wit_bindgen_rust::export!("../wit-files/rune/proc-block-v1.wit");

struct ProcBlockV1;

impl proc_block_v1::ProcBlockV1 for ProcBlockV1 {
    fn register_metadata() {
        let metadata = Metadata::new("FFT", env!("CARGO_PKG_VERSION"));
        metadata.set_description(
            "converts a signal from its original domain (often time or space) to a representation in the frequency domain.",
        );
        metadata.set_repository(env!("CARGO_PKG_REPOSITORY"));
        metadata.set_homepage(env!("CARGO_PKG_HOMEPAGE"));
        metadata.add_tag("stft");
        metadata.add_tag("frequency domain");

        let sampling_rate = ArgumentMetadata::new("Sample Rate");
        sampling_rate.set_description("Sampling Rate");
        sampling_rate.set_default_value("16000");
        let hint =
            runtime_v1::supported_argument_type(ArgumentType::UnsignedInteger);
        sampling_rate.add_hint(&hint);
        metadata.add_argument(&sampling_rate);

        let bins = ArgumentMetadata::new("Bins");
        bins.set_description("Intervals between samples in frequency domain");
        bins.set_default_value("480");
        let hint =
            runtime_v1::supported_argument_type(ArgumentType::UnsignedInteger);
        bins.add_hint(&hint);
        metadata.add_argument(&bins);

        let window_overlap = ArgumentMetadata::new("Window Overlap");
        window_overlap.set_description("Ratio of overlapped intervals.");
        window_overlap.set_default_value("0.6666667");
        let hint = runtime_v1::supported_argument_type(ArgumentType::Float);
        window_overlap.add_hint(&hint);
        metadata.add_argument(&window_overlap);

        let input = TensorMetadata::new("audio");
        input.set_description("A 1D tensor of `i16` samples.");
        let hint = supported_shapes(
            &[ElementType::I16],
            DimensionsParam::Fixed(&[1, 0]),
        );
        input.add_hint(&hint);
        metadata.add_input(&input);

        let output = TensorMetadata::new("output");
        output.set_description("output signal after applying STFT");
        let hint = supported_shapes(
            &[ElementType::U32],
            DimensionsParam::Fixed(&[1, 0]),
        );
        output.add_hint(&hint);
        metadata.add_output(&output);

        register_node(&metadata);
    }

    fn graph(node_id: String) -> Result<(), GraphError> {
        let ctx = GraphContext::for_node(&node_id)
            .ok_or(GraphError::MissingContext)?;

        ctx.add_input_tensor(
            "audio",
            ElementType::I16,
            DimensionsParam::Fixed(&[1, 0]),
        );
        ctx.add_output_tensor(
            "output",
            ElementType::F32,
            DimensionsParam::Fixed(&[1, 0]),
        );

        Ok(())
    }

    fn kernel(node_id: String) -> Result<(), KernelError> {
        let ctx = KernelContext::for_node(&node_id)
            .ok_or(KernelError::MissingContext)?;

        let sampling_rate = get_u32_args(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;
        let bins = get_u32_args(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;
        let window_overlap = get_f32_args(|n| ctx.get_argument(n))
            .map_err(KernelError::InvalidArgument)?;

        let TensorResult {
            element_type,
            dimensions,
            buffer,
        } = ctx.get_input_tensor("input").ok_or_else(|| {
            KernelError::InvalidInput(InvalidInput {
                name: "audio".to_string(),
                reason: BadInputReason::NotFound,
            })
        })?;

        check_input_dimensions(&dimensions);

        let input: Vec<i16> = buffer.elements().to_vec();

        let output = match element_type {
            ElementType::I16 => {
                transform_inner(input, sampling_rate, bins, window_overlap)
            },

            other => {
                return Err(KernelError::Other(format!(
                    "The FFT proc-block only accepts I16 tensors, found {:?}",
                    other,
                )))
            },
        };

        let output = match output {
            Some(ix) => ix,
            None => {
                return Err(KernelError::Other(
                    "The input tensor was empty".to_string(),
                ))
            },
        };

        let resulting_tensor = output.as_bytes();

        ctx.set_output_tensor(
            "output",
            TensorParam {
                element_type: ElementType::F32,
                dimensions: &dimensions,
                buffer: &resulting_tensor,
            },
        );

        Ok(())
    }
}

fn check_input_dimensions(dimensions: &[u32]) {
    assert_eq!(
        (!(dimensions.len() == 2 && dimensions[0] == 1)
            || !(dimensions.len() == 1)),
        true,
        "This proc block only supports 1D outputs (requested output: {:?})",
        dimensions
    );
}

fn get_u32_args(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<u32, InvalidArgument> {
    get_argument("count")
        .ok_or_else(|| InvalidArgument::not_found("count"))?
        .parse::<u32>()
        .map_err(|e| InvalidArgument::invalid_value("count", e))
}

fn get_f32_args(
    get_argument: impl FnOnce(&str) -> Option<String>,
) -> Result<f32, InvalidArgument> {
    get_argument("count")
        .ok_or_else(|| InvalidArgument::not_found("count"))?
        .parse::<f32>()
        .map_err(|e| InvalidArgument::invalid_value("count", e))
}

impl InvalidArgument {
    fn not_found(name: impl Into<String>) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::NotFound,
        }
    }

    fn invalid_value(name: impl Into<String>, reason: impl Display) -> Self {
        InvalidArgument {
            name: name.into(),
            reason: BadArgumentReason::InvalidValue(reason.to_string()),
        }
    }
}

fn transform_inner(
    input: Vec<i16>,
    sample_rate: u32,
    bins: u32,
    window_overlap: f32,
) -> Option<[u32; 1960]> {
    // Build the spectrogram computation engine
    let mut spectrograph = SpecOptionsBuilder::new(49, 241)
        .set_window_fn(sonogram::hann_function)
        .load_data_from_memory(input, sample_rate as u32)
        .build();

    // Compute the spectrogram giving the number of bins in a window and the
    // overlap between neighbour windows.
    spectrograph.compute(bins as usize, window_overlap);

    let spectrogram = spectrograph.create_in_memory(false);

    let filter_count: usize = 40;
    let power_spectrum_size = 241;
    let window_size = 480;
    let sample_rate_usize: usize = 16000;

    // build up the mel filter matrix
    let mut mel_filter_matrix =
        DMatrix::<f64>::zeros(filter_count, power_spectrum_size);
    for (row, col, coefficient) in mel::enumerate_mel_scaling_matrix(
        sample_rate_usize,
        window_size,
        power_spectrum_size,
        filter_count,
    ) {
        mel_filter_matrix[(row, col)] = coefficient;
    }

    let spectrogram = spectrogram.into_iter().map(f64::from);
    let power_spectrum_matrix_unflipped: DMatrix<f64> =
        DMatrix::from_iterator(49, power_spectrum_size, spectrogram);
    let power_spectrum_matrix_transposed =
        power_spectrum_matrix_unflipped.transpose();
    let mut power_spectrum_vec: Vec<_> =
        power_spectrum_matrix_transposed.row_iter().collect();
    power_spectrum_vec.reverse();
    let power_spectrum_matrix: DMatrix<f64> =
        DMatrix::from_rows(&power_spectrum_vec);
    let mel_spectrum_matrix = &mel_filter_matrix * &power_spectrum_matrix;
    let mel_spectrum_matrix = mel_spectrum_matrix.map(libm::sqrt);

    let min_value = mel_spectrum_matrix
        .data
        .as_vec()
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_value = mel_spectrum_matrix
        .data
        .as_vec()
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let res: Vec<u32> = mel_spectrum_matrix
        .data
        .as_vec()
        .iter()
        .map(|freq| 65536.0 * (freq - min_value) / (max_value - min_value))
        .map(|freq| freq as u32)
        .collect();

    let mut out = [0; 1960];
    out.copy_from_slice(&res[..1960]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let input = [0; 16000].to_vec();

        let got = transform_inner(input, 16000, 480, 0.6666667).unwrap();

        assert_eq!(got.len(), 1960);
    }
}
