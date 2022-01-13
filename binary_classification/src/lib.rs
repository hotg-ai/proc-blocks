#![no_std]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc-block which takes a rank 1 `tensor` as input, return 1 if value inside the tensor is greater than 1 otherwise 0.


#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct BinaryClassification {
    threshold: f32,
}

impl BinaryClassification {
    pub const fn new() -> Self {
        BinaryClassification {
            threshold: 0.5,
        }
    }
}

impl Default for BinaryClassification {
    fn default() -> Self {
        BinaryClassification::new()
    }
}

impl Transform<Tensor<f32>> for BinaryClassification {
    type Output = Tensor<u32>;

    fn transform(&mut self, input: Tensor<f32>) -> Tensor<u32> {
        // let value = input.into();
        let mut label: u32 = 0;
        if input.elements()> &[self.threshold] {
            label = 1
        }
        let v: Vec<u32> = vec![label];

        Tensor::new_vector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn test_binary_classification() {
        let v = Tensor::new_vector(vec![0.7]);
        let mut bin_class = BinaryClassification::default();
        let output =  bin_class.transform(v);
        let should_be: Tensor<u32> = [1].into();
        assert_eq!(output, should_be);
    }
}
