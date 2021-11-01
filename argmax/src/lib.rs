#![no_std]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Argmax {}

impl Argmax {
    pub const fn new() -> Self {
        Argmax {}
    }
}

impl Default for Argmax {
    fn default() -> Self {
        Argmax::new()
    }
}

impl Transform<Tensor<f32>> for Argmax {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        let (index, _) = input.elements().iter().enumerate().fold(
            (0, 0.0),
            |max, (ind, &val)| if val > max.1 { (ind, val) } else { max },
        );

        let v: Vec<u32> = vec![index as u32];

        Tensor::new_vector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn test_argmax() {
        let v = Tensor::new_vector(vec![2.3, 12.4, 55.1, 15.4]);
        let mut argmax = Argmax::default();
        let output = argmax.transform(v);
        let should_be: Tensor<u32> = [2].into();
        assert_eq!(output, should_be);
    }
}
