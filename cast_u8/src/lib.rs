#![no_std]

extern crate alloc;

use core::str;

use alloc::{borrow::Cow, vec::Vec};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct CastU8 {
    type: Vec<&'static str>
}

impl<T> Transform<Tensor<T>> for CastU8
where
    T: Copy + ToPrimitive,
{
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        let underlying_bytes: &[u8] = text.elements();
        let number_list =
            core::str::from_utf8(underlying_bytes).expect("Input tensor should be valid UTF8");
        
        let v: Vec<f32>= str::parse::<f32>(number_list);

        // if self.type == "u8"{
        //     let mut output: Vec<u8> = bytes.split("32").collect();
        //     output.iter_mut().map(|x|)

        // }

        Tensor::new_vector(v)
        
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn only_works_with_1d_inputs() {
        let mut proc_block = CastU8::default();
        let input: Tensor<i32> = Tensor::zeroed(alloc::vec![1, 2, 3]);

        let _ = proc_block.transform(input);
    }

    #[test]
    #[should_panic = "Index out of bounds: there are 2 labels but label 42 was requested"]
    fn label_index_out_of_bounds() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["first", "second"]);
        let input = Tensor::new_vector(alloc::vec![0_usize, 42]);

        let _ = proc_block.transform(input);
    }

    #[test]
    fn get_the_correct_labels() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![3, 1, 2]);
        let should_be = Tensor::new_vector(
            ["three", "one", "two"].iter().copied().map(Cow::Borrowed),
        );

        let got = proc_block.transform(input);

        assert_eq!(got, should_be);
    }
}