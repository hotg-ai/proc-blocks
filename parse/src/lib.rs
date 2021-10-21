#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
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
        let input = input.elements().iter();

        let trimmed_input = input.map(|s| trim_padded_string(s));
        let split_into_words =
            trimmed_input.flat_map(|line| line.split_whitespace());
        let parsed = split_into_words.map(parse);

        Tensor::new_vector(parsed)
    }
}

/// Remove the parts of the string after a trailing null
fn trim_padded_string(s: &str) -> &str {
    match s.find('\0') {
        Some(index) => &s[..index],
        None => s,
    }
}

fn parse<T>(input: &str) -> T
where
    T: FromStr,
    T::Err: Debug,
{
    input.parse().unwrap_or_else(|e| {
        panic!(
            "Unable to parse {:?} as a {}: {:?}",
            input,
            core::any::type_name::<T>(),
            e
        );
    })
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
