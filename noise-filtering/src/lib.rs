#![no_std]

#[macro_use]
extern crate alloc;

mod gain_control;
mod noise_reduction;

pub use crate::noise_reduction::ScaledU16;

use crate::{gain_control::GainControl, noise_reduction::NoiseReduction};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, ProcBlock)]
pub struct NoiseFiltering {
    // gain control options
    strength: f32,
    offset: f32,
    gain_bits: i32,
    #[proc_block(skip)]
    gain_control: gain_control::State,

    // noise filtering options
    smoothing_bits: u32,
    even_smoothing: ScaledU16,
    odd_smoothing: ScaledU16,
    min_signal_remaining: ScaledU16,
    #[proc_block(skip)]
    noise_reduction: noise_reduction::State,
}

impl Transform<Tensor<u32>> for NoiseFiltering {
    type Output = Tensor<i8>;

    fn transform(&mut self, input: Tensor<u32>) -> Tensor<i8> {
        let NoiseFiltering {
            strength,
            offset,
            gain_bits,
            ref mut gain_control,
            smoothing_bits,
            even_smoothing,
            odd_smoothing,
            min_signal_remaining,
            ref mut noise_reduction,
        } = *self;

        let n = NoiseReduction {
            even_smoothing,
            min_signal_remaining,
            odd_smoothing,
            smoothing_bits,
        };
        let cleaned = n.transform(input, noise_reduction);

        let g = GainControl {
            gain_bits,
            offset,
            strength,
        };
        let amplified = g
            .transform(
                cleaned,
                &noise_reduction.estimate,
                smoothing_bits as u16,
                gain_control,
            )
            .map(|_, energy| libm::log2((*energy as f64) + 1.0));

        let (min_value, max_value) = amplified.elements().iter().copied().fold(
            (f64::NEG_INFINITY, f64::INFINITY),
            |(lower, upper), current| (lower.min(current), upper.max(current)),
        );

        amplified.map(|_, energy| {
            ((255.0 * (energy - min_value) / (max_value - min_value)) - 128.0)
                as i8
        })
    }
}

impl Default for NoiseFiltering {
    fn default() -> Self {
        let NoiseReduction {
            smoothing_bits,
            even_smoothing,
            odd_smoothing,
            min_signal_remaining,
        } = NoiseReduction::default();
        let config = GainControl::default();
        let GainControl {
            strength,
            offset,
            gain_bits,
        } = config;

        NoiseFiltering {
            strength,
            offset,
            gain_bits,
            gain_control: gain_control::State::new(
                config,
                smoothing_bits as u16,
            ),
            smoothing_bits,
            even_smoothing,
            odd_smoothing,
            min_signal_remaining,
            noise_reduction: noise_reduction::State::default(),
        }
    }
}
