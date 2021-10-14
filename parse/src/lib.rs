#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::Debug;
use core::str::FromStr;
use alloc::borrow::Cow;
use core::iter::IntoIterator;
use core::marker::PhantomData;
use hotg_rune_proc_blocks::{ProcBlock, Tensor, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
pub struct Parse<T> {
    _output_type: PhantomData<T>,
}

impl<T> Default for Parse<T> {
    fn default() -> Self {
        Parse {
            _output_type: PhantomData,
        }
    }
}

impl<T> Transform<Tensor<Cow<'static, str>>> for Parse<T>
where
    T: Copy + Debug + FromStr,
    T::Err: Debug,
{
    type Output = Tensor<T>;

    fn transform(&mut self, input: Tensor<Cow<'static, str>>) -> Self::Output {

        let number_list: Vec<&str> = input.lines();
        let mut v = Vec::new();

        for i in number_list.into_iter() {
            let val = T::from_str(i);
            match val {
                Ok(value) => value,
                Err(e) => panic!(
                    "Unable to parse \"{}\" as a {}: {:?}",
                    input,
                    core::any::type_name::<T>(),
                    e
                ),
            }
            v.push(val);
        }

        // let v = input.split("\n")
        // .map(|s| s.to_owned()
        //     .parse::<T>()
        //     .map_err(|e| e.to_string())
        // )
        // .collect();

        Tensor::new_vector(v)
        
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use alloc::vec;
//     #[test]
//     fn test_for_f32() {
//         let mut cast = Parse::default();
//         let bytes: Vec<u8> = vec!["-45.0\n1.0"]; // u8 coding of [-45, 1]
//         let input = Tensor::new_vector(bytes);

//         let output = cast.transform(input);

//         let should_be = Tensor::new_vector(vec![-45.0, 1.0]);

//         assert_eq!(output, should_be);
//     }
// }
