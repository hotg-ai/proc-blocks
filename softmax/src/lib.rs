#![no_std]

extern crate alloc;

use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};
use libm::expf;

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Softmax {}

impl Softmax {
    pub const fn new() -> Self {
        Softmax {}
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Softmax::new()
    }
}

impl Transform<Tensor<f32>> for Softmax {
    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let b = input.map(|_, &x| expf(x as f32));
        let sum: f32 = b.elements().iter().sum();

        b.map(|_, &x| x / sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn test_softmax() {
        let v = Tensor::new_vector(vec![2.3, 12.4, 5.1]);
        let mut softmax = Softmax::default();
        let output = softmax.transform(v);
        let should_be =
            Tensor::new_vector(vec![0.000041050153, 0.99928397, 0.00067505526]);
        assert_eq!(output, should_be);
    }
}
