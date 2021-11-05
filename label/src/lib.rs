#![no_std]

pub mod into_index_macro;
pub use into_index_macro::IntoIndex;

extern crate alloc;

use alloc::{borrow::Cow, vec::Vec};
use core::fmt::Debug;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc block which, when given a set of indices, will return their
/// associated labels.
///
/// # Examples
///
/// ```rust
/// # use label::Label;
/// # use hotg_rune_proc_blocks::{Transform, Tensor};
/// let mut proc_block = Label::default();
/// proc_block.set_labels(["zero", "one", "two", "three"]);
/// let input = Tensor::new_vector(vec![3, 1, 2]);
///
/// let got = proc_block.transform(input);
///
/// assert_eq!(got.elements(), &["three", "one", "two"]);

#[derive(Debug, Default, Clone, PartialEq, ProcBlock)]
pub struct Label {
    labels: Vec<&'static str>,
}

impl Label {
    pub fn set_wordlist(&mut self, wordlist: &'static str) {
        self.labels = wordlist.lines().map(|l| l.trim()).collect();
    }

    fn get_by_index(&mut self, ix: usize) -> Cow<'static, str> {
        // Note: We use a more cumbersome match statement instead of unwrap()
        // to provide the user with more useful error messages
        match self.labels.get(ix) {
            Some(&label) => label.into(),
            None => panic!("Index out of bounds: there are {} labels but label {} was requested", self.labels.len(), ix)
        }
    }
}

impl<T> Transform<Tensor<T>> for Label
where
    T: Copy + IntoIndex,
{
    type Output = Tensor<Cow<'static, str>>;

    fn transform(&mut self, input: Tensor<T>) -> Self::Output {
        let indices = input
            .elements()
            .iter()
            .copied()
            .map(IntoIndex::try_into_index);

        indices.map(|ix| self.get_by_index(ix)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_the_correct_labels() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![2, 0, 1]);
        let should_be = Tensor::new_vector(
            ["two", "zero", "one"].iter().copied().map(Cow::Borrowed),
        );

        let got = proc_block.transform(input);

        assert_eq!(got, should_be);
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
    #[should_panic = "UNSUPPORTED: Can't be converted to usize. It only supports u8, u16, u32, u64, i32, i64 ( with positive numbers) f32 (with their fractional part zero E.g. 2.0, 4.0, etc)"]
    fn get_the_correct_labels_panic() {
        let mut proc_block = Label::default();
        proc_block.set_labels(["zero", "one", "two", "three"]);
        let input = Tensor::new_vector(alloc::vec![-3, -1, -2]);

        let _got = proc_block.transform(input);
    }
}
