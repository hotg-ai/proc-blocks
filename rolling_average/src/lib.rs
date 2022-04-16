#![cfg_attr(not(feature = "metadata"), no_std)]

extern crate alloc;

use alloc::vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct RollingAverage {}

impl RollingAverage {
    pub const fn new() -> Self { RollingAverage
     {} }
}

impl Default for RollingAverage {
    fn default() -> Self { RollingAverage
    ::new() }
}

impl Transform<(Tensor<f32>, Tensor<u32>, Tensor<f32>)> for RollingAverage {
    type Output = (Tensor<f32>, Tensor<u32>);
    fn transform(&mut self, inputs:(Tensor<f32>, Tensor<u32>, Tensor<f32>)) -> (Tensor<f32>, Tensor<u32>) {
        let (previous_average, k, sample) = inputs;
        let result = vec!((previous_average.elements()[0]*(((k.elements()[0]-1) as f32)/(k.elements()[0] as f32))) + (sample.elements()[0] / (k.elements()[0] as f32)));
        let x = vec!(k.elements()[0] + 1);
        return (Tensor::new_vector(result),Tensor::new_vector(x));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_rolling_average
() {
        let previous_average: Tensor<f32> = Tensor::new_vector(vec![5.0]);
        let k: Tensor<u32> = Tensor::new_vector(vec![1]);
        let sample: Tensor<f32> = Tensor::new_vector(vec![5.0]);
        let mut rolling_average = RollingAverage::default();
        let input = (previous_average, k, sample);
        let output = rolling_average.transform(input);
        let new_average_should_be =
            Tensor::new_vector(vec![5.0]);
        let new_k_should_be: Tensor<u32> = [2].into();
        assert_eq!(output, (new_average_should_be, new_k_should_be));
    }
}
