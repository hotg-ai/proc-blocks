//! A gain control routine ported from the [TensorFlow function][tf].
//!
//! [tf]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c

use alloc::vec::Vec;

pub const WIDE_DYNAMIC_FUNCTION_BITS: usize = 32;
pub const WIDE_DYNAMIC_FUNCTION_LUT_SIZE: usize =
    4 * WIDE_DYNAMIC_FUNCTION_BITS - 3;
pub const PCAN_SNR_BITS: i32 = 12;
pub const PCAN_OUTPUT_BITS: usize = 6;
pub const SMOOTHING_BITS: u16 = 10;
pub const CORRECTION_BITS: i32 = -1;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GainControl {
    pub strength: f32,
    pub offset: f32,
    pub gain_bits: i32,
}

impl GainControl {
    pub(crate) fn transform<'a>(
        &'a self,
        mut input: &'a mut [u32],
        noise_estimate: &[u32],
        smoothing_bits: u16,
        state: &'a mut State,
    ) -> &[u32] {
        state.update(*self, smoothing_bits);
        state.transform(&mut input, noise_estimate)
    }
}

impl Default for GainControl {
    fn default() -> Self {
        GainControl {
            strength: 0.95,
            offset: 80.0,
            gain_bits: 21,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    gain_lut: Vec<i16>,
    snr_shift: i32,
    config: GainControl,
}

impl State {
    pub(crate) fn new(config: GainControl, smoothing_bits: u16) -> Self {
        let mut gain_lut = vec![0; WIDE_DYNAMIC_FUNCTION_LUT_SIZE];
        let snr_shift = config.gain_bits - CORRECTION_BITS - PCAN_SNR_BITS;
        let input_bits = smoothing_bits as i32 - CORRECTION_BITS;

        gain_lut[0] = gain_lookup(config, input_bits, 0);
        gain_lut[1] = gain_lookup(config, input_bits, 1);

        for interval in 2..=WIDE_DYNAMIC_FUNCTION_BITS {
            let x_0: u32 = 1_u32 << (interval - 1);
            let x_1 = x_0 + (x_0 >> 1);
            let x_2 = if interval == WIDE_DYNAMIC_FUNCTION_BITS {
                x_0 + (x_0 - 1)
            } else {
                2 * x_0
            };

            let y_0 = gain_lookup(config, input_bits, x_0);
            let y_1 = gain_lookup(config, input_bits, x_1);
            let y_2 = gain_lookup(config, input_bits, x_2);

            let diff_1 = y_1 - y_0;
            let diff_2 = y_2 - y_0;
            let a_1 = 4 * diff_1 - diff_2;
            let a_2 = diff_2 - a_1;

            gain_lut[4 * interval - 6] = y_0;
            gain_lut[4 * interval - 6 + 1] = a_1;
            gain_lut[4 * interval - 6 + 2] = a_2;
        }

        State {
            gain_lut,
            snr_shift,
            config,
        }
    }

    pub fn update(&mut self, config: GainControl, smoothing_bits: u16) {
        if self.config != config {
            *self = State::new(config, smoothing_bits);
        }
    }

    pub fn transform<'a>(
        &'a mut self,
        mut input: &'a mut [u32],
        noise_estimate: &[u32],
    ) -> &[u32] {
        let elements = input.as_mut();

        for (i, element) in elements.iter_mut().enumerate() {
            let gain =
                wide_dynamic_function(noise_estimate[i], &self.gain_lut) as u32;
            let signal = *element;
            let snr = (signal as u64 * gain as u64) >> self.snr_shift;
            *element = shrink(snr as u32);
        }

        input
    }
}

fn shrink(snr: u32) -> u32 {
    if snr < (2_u32 << PCAN_SNR_BITS) {
        snr.wrapping_mul(snr)
            >> (2 + 2 * PCAN_SNR_BITS - PCAN_OUTPUT_BITS as i32)
    } else {
        (snr >> (PCAN_SNR_BITS - PCAN_OUTPUT_BITS as i32))
            .wrapping_sub(1 << PCAN_OUTPUT_BITS as i32)
    }
}

fn most_significant_bit(number: u32) -> usize {
    32 - number.leading_zeros() as usize
}

fn wide_dynamic_function(x: u32, lookup_table: &[i16]) -> i16 {
    if x <= 2 {
        return lookup_table[x as usize];
    }

    let interval = most_significant_bit(x) as i16;

    let index_offset = 4 * interval as usize - 6;

    let frac = if interval < 11 {
        x << (11 - interval)
    } else {
        x >> (interval - 11)
    };
    let frac = (frac & 0x3ff) as i16;

    let mut result = (lookup_table[index_offset + 2] as i32 * frac as i32) >> 5;
    result += ((lookup_table[index_offset + 1] as u32) << 5) as i32;
    result *= frac as i32;
    result = (result + (1_i32 << 14)) >> 15;
    result += lookup_table[index_offset] as i32;

    result as i16
}

fn gain_lookup(config: GainControl, input_bits: i32, x: u32) -> i16 {
    let x = (x as f32) / (1 << input_bits) as f32;
    let gain = (1 << config.gain_bits) as f32
        * libm::powf(x + config.offset, -config.strength);

    let gain = f32::min(gain, i16::max_value() as f32);

    (gain + 0.5) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// https://github.com/tensorflow/tensorflow/blob/0f6d728b920e9b0286171bdfec9917d8486ac08b/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_test.cc#L43-L63
    #[test]
    fn test_pcan_gain_control() {
        let gain_control = GainControl {
            strength: 0.95,
            offset: 80.0,
            ..Default::default()
        };
        let input = vec![241137, 478104];
        // Note: we get this from a the noise reduction step
        let noise_estimate = vec![6321887, 31248341];
        let mut state = State::new(gain_control, SMOOTHING_BITS);

        let got = state.transform(&mut input, &noise_estimate);

        let should_be = vec![3578, 1533];
        assert_eq!(got, should_be);
    }

    #[test]
    fn initialize_state() {
        let config = GainControl {
            strength: 0.95,
            offset: 80.0,
            gain_bits: 21,
        };

        let got = State::new(config, SMOOTHING_BITS);

        let should_be = State {
            snr_shift: 10,
            gain_lut: vec![
                32636, 32636, 32635, 0, 0, 0, 32635, 1, -2, 0, 32634, 1, -2, 0,
                32633, -5, 2, 0, 32630, -6, 0, 0, 32624, -12, 0, 0, 32612, -23,
                -2, 0, 32587, -48, 0, 0, 32539, -96, 0, 0, 32443, -190, 0, 0,
                32253, -378, 4, 0, 31879, -739, 18, 0, 31158, -1409, 62, 0,
                29811, -2567, 202, 0, 27446, -4301, 562, 0, 23707, -6265, 1230,
                0, 18672, -7458, 1952, 0, 13166, -7030, 2212, 0, 8348, -5342,
                1868, 0, 4874, -3459, 1282, 0, 2697, -2025, 774, 0, 1446,
                -1120, 436, 0, 762, -596, 232, 0, 398, -313, 122, 0, 207, -164,
                64, 0, 107, -85, 34, 0, 56, -45, 18, 0, 29, -22, 8, 0, 15, -13,
                6, 0, 8, -8, 4, 0, 4, -2, 0,
            ],
            config,
        };

        assert_eq!(got, should_be);
    }

    #[test]
    fn known_wide_dynamic_function_results() {
        let config = GainControl {
            strength: 0.95,
            offset: 80.0,
            gain_bits: 21,
        };
        let state = State::new(config, SMOOTHING_BITS);

        let inputs = vec![(6321887, 990), (31248341, 219)];

        for (input, should_be) in inputs {
            let got = wide_dynamic_function(input, &state.gain_lut);
            assert_eq!(got, should_be);
        }
    }
}
