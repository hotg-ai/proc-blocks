#![no_std]

extern crate alloc;

use alloc::{borrow::Cow, vec::Vec};
use core::{fmt::Debug, marker::PhantomData, str::FromStr};
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

/// A proc block which can parse a string to numbers.

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
        let number_list = &input;

        let mut w = Vec::new();

        // split at `\n`
        for s in number_list.into_iter() {
            let mut st = &s[..s.len()];
            if let Some(index) = s.find("\u{0000}") {
                st = &s[..index];
            }
            let x: Vec<&str> = st.lines().collect();
            w.extend(x);
        }

        let mut u = Vec::new();

        // split at whitespaces.
        for s in w.into_iter() {
            let x: Vec<&str> = s.split_whitespace().collect();
            u.extend(x);
        }

        let mut v = Vec::new();

        for i in u.into_iter() {
            let val = T::from_str(i);
            match val {
                Ok(value) => v.push(value),
                Err(e) => panic!(
                    "Unable to parse \"{}\" as a {}: {:?}",
                    i,
                    core::any::type_name::<T>(),
                    e
                ),
            };
        }

        Tensor::new_vector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_for_number_in_lines() {
        let mut parser = Parse::default();
        let bytes = vec![Cow::Borrowed("5\n6\n7")];
        let input = Tensor::new_vector(bytes);
        let output = parser.transform(input);
        let should_be = Tensor::new_vector(vec![5, 6, 7]);
        assert_eq!(output, should_be);
    }

    #[test]
    fn test_for_number_in_vec() {
        let mut parser = Parse::default();
        let bytes =
            vec![Cow::Borrowed("5"), Cow::Borrowed("6"), Cow::Borrowed("7")];
        let input = Tensor::new_vector(bytes);
        let output = parser.transform(input);
        let should_be = Tensor::new_vector(vec![5, 6, 7]);
        assert_eq!(output, should_be);
    }
    #[test]
    fn test_for_number_in_lines_and_vec() {
        let mut parser = Parse::default();
        let bytes = vec![
            Cow::Borrowed("2"),
            Cow::Borrowed("3"),
            Cow::Borrowed("4"),
            Cow::Borrowed("5\n6\n7"),
        ];
        let input = Tensor::new_vector(bytes);
        let output = parser.transform(input);
        let should_be = Tensor::new_vector(vec![2, 3, 4, 5, 6, 7]);
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
    #[should_panic(
        expected = "Unable to parse \"a\" as a f64: ParseFloatError { kind: Invalid }"
    )]
    fn test_for_invalid_data_type() {
        let mut cast = Parse::default();
        let bytes = vec![Cow::Borrowed("1.0\na")];
        let input = Tensor::new_vector(bytes);
        let output = cast.transform(input);
        let should_be = Tensor::new_vector(vec![1.0, 2.0]);
        assert_eq!(output, should_be);
    }
}
