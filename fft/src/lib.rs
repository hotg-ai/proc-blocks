#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

use hotg_rune_proc_blocks::{
    guest::{
        parse, Argument, ArgumentMetadata, ArgumentType, CreateError,
        ElementType, Metadata, ProcBlock, RunError, Tensor, TensorConstraint,
        TensorConstraints, TensorMetadata,
    },
    ndarray::Array1,
};
use nalgebra::DMatrix;
use sonogram::SpecOptionsBuilder;

hotg_rune_proc_blocks::export_proc_block! {
    metadata: metadata,
    proc_block: Fft,
}

fn metadata() -> Metadata {
    Metadata::new("FFT", env!("CARGO_PKG_VERSION"))
        .with_description(
            "converts a signal from its original domain (often time or space) to a representation in the frequency domain.",
        )
        .with_repository(env!("CARGO_PKG_REPOSITORY"))
        .with_homepage(env!("CARGO_PKG_HOMEPAGE"))
        .with_tag("stft")
        .with_tag("frequency domain")
        .with_argument(
            ArgumentMetadata::new("sample_rate")
                .with_description("Sampling rate")
                .with_default_value("16000")
                .with_hint(ArgumentType::UnsignedInteger)
        )
        .with_argument(
            ArgumentMetadata::new("bins")
                .with_description("Intervals between samples in frequency domain")
                .with_default_value("480")
                .with_hint(ArgumentType::UnsignedInteger)
        )
        .with_argument(
            ArgumentMetadata::new("window_overlap")
                .with_description("Ratio of overlapped intervals.")
                .with_default_value("0.6666667")
                .with_hint(ArgumentType::Float)
        )
        .with_input(
            TensorMetadata::new("audio")
                .with_description("A 1D tensor containing PCM-encoded audio samples.")
        )
        .with_output(
            TensorMetadata::new("output")
                .with_description("output signal after applying STFT")
        )
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Fft {
    sample_rate: u32,
    bins: u32,
    window_overlap: f32,
}

impl ProcBlock for Fft {
    fn tensor_constraints(&self) -> TensorConstraints {
        TensorConstraints {
            inputs: vec![TensorConstraint::new(
                "audio",
                ElementType::I16,
                [1, 0],
            )],
            outputs: vec![TensorConstraint::new(
                "output",
                ElementType::F32,
                [1, 0],
            )],
        }
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, RunError> {
        let input = Tensor::get_named(&inputs, "audio")?.view_1d()?;

        let output = transform_inner(
            input.to_vec(),
            self.sample_rate,
            self.bins,
            self.window_overlap,
        );

        Ok(vec![Tensor::new("output", &output)])
    }
}

impl TryFrom<Vec<Argument>> for Fft {
    type Error = CreateError;

    fn try_from(args: Vec<Argument>) -> Result<Self, Self::Error> {
        let sample_rate =
            parse::optional_arg(&args, "sample_rate")?.unwrap_or(16000);
        let bins = parse::optional_arg(&args, "bins")?.unwrap_or(480);
        let window_overlap =
            parse::optional_arg(&args, "window_overlap")?.unwrap_or(0.6666667);

        Ok(Fft {
            sample_rate,
            bins,
            window_overlap,
        })
    }
}

fn transform_inner(
    input: Vec<i16>,
    sample_rate: u32,
    bins: u32,
    window_overlap: f32,
) -> Array1<u32> {
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
    let mel_spectrum_matrix = mel_spectrum_matrix.map(f64::sqrt);

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

    mel_spectrum_matrix
        .data
        .as_vec()
        .iter()
        .map(|freq| 65536.0 * (freq - min_value) / (max_value - min_value))
        .map(|freq| freq as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let input = [0; 16000].to_vec();

        let got = transform_inner(input, 16000, 480, 0.6666667);

        assert_eq!(got.len(), 1960);
    }
}
