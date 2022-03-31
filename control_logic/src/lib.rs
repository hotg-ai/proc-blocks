#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock, Default)]
pub struct ControlLogic {}

impl Transform<(Tensor<u32>, Tensor<u32>)> for ControlLogic {
    type Output = Tensor<u32>;
    fn transform(&mut self, inputs: (Tensor<u32>, Tensor<u32>)) -> Self::Output {
        let mut output: Vec<u32> = Vec::new();
        let (input1, input2) = inputs;
        match input1.elements()[0] {
            1 => output.push(input2.elements()[0]),
            _ => output.push(6),
        }
        Tensor::new_vector(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_logic() {
        let mut logic = ControlLogic::default();
        let result = logic.transform(([0].into(), [1].into()));
        let should_be: Tensor<u32> = [1].into();
        assert_eq!(result, should_be)
    }
}
