//! A noise reduction routine inspired by the [TensorFlow function][tf].
//!
//! [tf]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c

use std::str::FromStr;

const NOISE_REDUCTION_BITS: usize = 14;

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct State {
    pub estimate: Vec<u32>,
}

/// Core logic for the noise reduction step.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseReduction {
    pub smoothing_bits: u32,
    pub even_smoothing: ScaledU16,
    pub odd_smoothing: ScaledU16,
    pub min_signal_remaining: ScaledU16,
}

impl NoiseReduction {
    pub(crate) fn transform<'a>(
        &'a self,
        input: &'a mut [u32],
        state: &mut State,
    ) -> &mut [u32] {
        // make sure we have the right estimate buffer size and panic if we
        // don't. This works because the input and output have the same
        // dimensions.
        match [1, input.len()] {
            [1, len, ] => state.estimate.resize(len, 0),
            other => panic!(
                "This transform only supports outputs of the form [1, _], not {:?}",
                other
            ),
        }

        let signal = input.as_mut();

        for (i, value) in signal.iter_mut().enumerate() {
            let smoothing = if i % 2 == 0 {
                self.even_smoothing.0 as u64
            } else {
                self.odd_smoothing.0 as u64
            };

            let one_minus_smoothing = 1 << NOISE_REDUCTION_BITS;

            // update the estimate of the noise
            let signal_scaled_up = (*value << self.smoothing_bits) as u64;
            let mut estimate = ((signal_scaled_up * smoothing)
                + (state.estimate[i] as u64 * one_minus_smoothing))
                >> NOISE_REDUCTION_BITS;
            state.estimate[i] = estimate as u32;

            // Make sure that we can't get a negative value for the signal
            // estimate
            estimate = core::cmp::min(estimate, signal_scaled_up);

            let floor = (*value as u64 * self.min_signal_remaining.0 as u64)
                >> NOISE_REDUCTION_BITS;
            let subtracted =
                (signal_scaled_up - estimate) >> self.smoothing_bits;

            *value = core::cmp::max(floor, subtracted) as u32;
        }

        input
    }
}

impl Default for NoiseReduction {
    fn default() -> Self {
        NoiseReduction {
            smoothing_bits: crate::gain_control::SMOOTHING_BITS.into(),
            even_smoothing: ScaledU16::from(0.025),
            odd_smoothing: ScaledU16::from(0.06),
            min_signal_remaining: ScaledU16::from(0.05),
        }
    }
}

/// A `u16` which can be parsed from a float that gets scaled from `[0, 1]` to
/// `[0, 2^14]`.
///
/// # Examples
///
/// ```
/// # use noise_filtering::ScaledU16;
/// let input = "0.5";
/// let parsed: ScaledU16 = input.parse().unwrap();
/// assert_eq!(parsed.0, (1 << 14)/2);
/// ```
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScaledU16(pub u16);

impl From<f32> for ScaledU16 {
    fn from(number: f32) -> Self {
        let scale_factor: f32 = (1 << NOISE_REDUCTION_BITS) as f32;
        ScaledU16((number.clamp(0.0, 1.0) * scale_factor) as u16)
    }
}

impl From<ScaledU16> for f32 {
    fn from(scaled: ScaledU16) -> Self {
        let scale_factor: f32 = (1 << NOISE_REDUCTION_BITS) as f32;
        scaled.0 as f32 / scale_factor
    }
}

impl FromStr for ScaledU16 {
    type Err = core::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let float: f32 = s.parse()?;
        Ok(ScaledU16::from(float))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_test.cc#L41-L59
    #[test]
    fn test_noise_reduction_estimate() {
        let noise_reduction = NoiseReduction::default();
        let mut input = vec![247311, 508620];
        let should_be = vec![6321887, 31248341];
        let mut state = State::default();

        let _ = noise_reduction.transform(&mut input, &mut state);

        assert_eq!(state.estimate, should_be);
    }

    /// https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_test.cc#L61-L79
    #[test]
    fn test_noise_reduction() {
        let noise_reduction = NoiseReduction::default();
        let mut input = vec![247311, 508620];
        let should_be = vec![241137, 478104];
        let mut state = State::default();

        let got = noise_reduction.transform(&mut input, &mut state);

        assert_eq!(got, should_be);
    }
}
