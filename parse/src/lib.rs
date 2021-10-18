#![no_std]

extern crate alloc;

use crate::alloc::string::ToString;
use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::IntoIterator;
use core::marker::PhantomData;
use core::str::FromStr;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc block which can parse a string to numbers.
/// To get [1,2,3,4,5] as output from this proc-block, send your input to RAW capability in this format:
/// 1
/// 2
/// 3
/// 4
/// 5

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Parse<T: 'static> {
    #[proc_block(skip)]
    _output_type: PhantomData<T>,
}

impl<T: 'static> Default for Parse<T> {
    fn default() -> Self {
        Parse {
            _output_type: PhantomData,
        }
    }
}

impl<T> Transform<Tensor<Cow<'static, str>>> for Parse<T>
where
    T: Copy + Debug + FromStr + 'static,
    T::Err: Debug,
{
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<Cow<'static, str>>) -> Self::Output {
        let input = input.elements();
        let number_list = input[0].to_string();
        let mut index = number_list.len();
        if number_list.contains("\u{0000}") {
            index = number_list.find("\u{0000}").unwrap();
        }
        let number_list = &number_list[..index];
        let w: Vec<&str> = number_list.lines().collect();

        let mut v = Vec::new();

        for i in w.into_iter() {
            let val = T::from_str(i);
            match val {
                Ok(value) => value,
                Err(e) => panic!(
                    "Unable to parse \"{}\" as a {}: {:?}",
                    i,
                    core::any::type_name::<T>(),
                    e
                ),
            };
            v.push(val.unwrap());
        }

        Tensor::new_vector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_for_number() {
        let mut parser = Parse::default();
        let bytes = vec![Cow::Borrowed("5\n6\n7")];
        let input = Tensor::new_vector(bytes);
        let output = parser.transform(input);
        let should_be = Tensor::new_vector(vec![5, 6, 7]);
        assert_eq!(output, should_be);
    }

    #[test]
    fn test() {
        let mut parser = Parse::default();
        let bytes = vec![Cow::Borrowed("5\n6\n7\u{0000}")];
        let input = Tensor::new_vector(bytes);
        let output = parser.transform(input);
        let should_be = Tensor::new_vector(vec![5, 6, 7]);
        assert_eq!(output, should_be);
    }

    #[test]
    #[should_panic]
    fn test_for_invalid_data_type() {
        let mut cast = Parse::default();
        let bytes = vec![Cow::Borrowed("1.0\na")];
        let input = Tensor::new_vector(bytes);
        let output = cast.transform(input);
        let should_be = Tensor::new_vector(vec![1.0, 2.0]);
        assert_eq!(output, should_be);
    }
}
