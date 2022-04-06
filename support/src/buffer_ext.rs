use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, ErrorKind, ShapeError};

use crate::ValueType;

/// Extension traits added to a byte buffer.
pub trait BufferExt {
    /// Reinterpret this byte buffer as a slice of `T`'s.
    fn elements<T: ValueType>(&self) -> &[T];
    /// Reinterpret this byte buffer as a mutable slice of `T`'s.
    fn elements_mut<T: ValueType>(&mut self) -> &mut [T];

    /// Interpret this buffer as a sequence of UTF-8 strings, where each string
    /// is prefixed by its length as a little-endian `u16`.
    fn strings(&self) -> Result<Vec<&str>, ShapeError>;

    /// View the buffer as a multi-dimensional array.
    fn view<T: ValueType>(
        &self,
        dimensions: &[u32],
    ) -> Result<ArrayViewD<'_, T>, ShapeError> {
        let elements = self.elements();
        let dimensions = dims(dimensions);
        ArrayViewD::from_shape(dimensions, elements)
    }

    /// View the buffer as a mutable multi-dimensional array.
    fn view_mut<T: ValueType>(
        &mut self,
        dimensions: &[u32],
    ) -> Result<ArrayViewMutD<'_, T>, ShapeError> {
        let elements = self.elements_mut();
        let dimensions = dims(dimensions);
        ArrayViewMutD::from_shape(dimensions, elements)
    }

    fn string_view<'a>(
        &'a self,
        dimensions: &[u32],
    ) -> Result<ArrayD<&'a str>, ShapeError> {
        let strings = self.strings()?;
        let dimensions = dims(dimensions);
        ArrayD::from_shape_vec(dimensions, strings)
    }
}

fn dims(d: &[u32]) -> Vec<usize> {
    d.iter()
        .map(|&dim| usize::try_from(dim).expect("Conversion should never fail"))
        .collect()
}

impl BufferExt for [u8] {
    fn elements<T: ValueType>(&self) -> &[T] {
        unsafe {
            let (start, middle, end) = self.align_to::<T>();
            assert!(start.is_empty());
            assert!(end.is_empty());
            middle
        }
    }

    fn elements_mut<T: ValueType>(&mut self) -> &mut [T] {
        unsafe {
            let (start, middle, end) = self.align_to_mut::<T>();
            assert!(start.is_empty());
            assert!(end.is_empty());
            middle
        }
    }

    fn strings(&self) -> Result<Vec<&str>, ShapeError> {
        const HEADER_SIZE: usize = std::mem::size_of::<u32>();

        let mut strings = Vec::new();
        let mut buffer = self;

        while !buffer.is_empty() {
            if buffer.len() < HEADER_SIZE {
                // We don't have enough bytes remaining for a full length field,
                // so something is probably wrong with our buffer.
                return Err(ShapeError::from_kind(ErrorKind::OutOfBounds));
            }

            let (len, rest) = buffer.split_at(HEADER_SIZE);

            let len: [u8; HEADER_SIZE] = len.try_into().expect("Unreachable");
            let len = u32::from_le_bytes(len);
            let len = usize::try_from(len).expect("Unreachable");

            if rest.len() < len {
                // We don't have enough bytes left in the buffer to read a
                // string with this length.
                return Err(ShapeError::from_kind(ErrorKind::OutOfBounds));
            }

            let (s, rest) = rest.split_at(len);

            match std::str::from_utf8(s) {
                Ok(s) => strings.push(s),
                Err(_) => {
                    // The string wasn't valid UTF-8. We're probably using the
                    // wrong ShapeError here, but our alternative would be
                    // introducing our own error type and that seems overkill.
                    return Err(ShapeError::from_kind(
                        ErrorKind::IncompatibleLayout,
                    ));
                },
            }

            buffer = rest;
        }

        Ok(strings)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use ndarray::ErrorKind;

    use super::*;

    fn as_byte_buffer_mut<T: ValueType>(items: &mut [T]) -> &mut [u8] {
        // Safety: Invariant upheld by the ValueType impl
        let (head, bytes, tail) = unsafe { items.align_to_mut() };
        assert!(head.is_empty());
        assert!(tail.is_empty());
        bytes
    }

    #[test]
    fn view_4_floats_as_2x2() {
        let floats = &[0.0_f32, 1.0, 2.0, 3.0];
        let buffer: Vec<u8> =
            floats.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let dimensions = &[2, 2];

        let tensor = buffer.view::<f32>(dimensions).unwrap();

        assert_eq!(tensor.dim(), ndarray::Dim(vec![2, 2]));
        assert_eq!(tensor[[0, 0]], 0.0);
        assert_eq!(tensor[[0, 1]], 1.0);
        assert_eq!(tensor[[1, 0]], 2.0);
        assert_eq!(tensor[[1, 1]], 3.0);
    }

    #[test]
    fn incorrect_size_is_error() {
        let buffer = [1_u8, 2, 3, 4];
        let dimensions = &[5];

        let error = buffer.view::<u8>(dimensions).unwrap_err();

        let kind = error.kind();
        assert_eq!(kind, ErrorKind::OutOfBounds);
    }

    #[test]
    fn mutate_tensor_in_place() {
        let mut floats = [0.0_f32, 0.0, 0.0, 0.0];
        let dimensions = &[2, 2];

        {
            let buffer = as_byte_buffer_mut(&mut floats);
            let mut tensor = buffer.view_mut::<f32>(dimensions).unwrap();

            tensor[[1, 0]] = 5.0;
        };

        assert_eq!(floats, [0.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn load_string_tensor() {
        let strings = ["this", "is a", "sentence", "."];
        let mut buffer = Vec::new();
        for s in &strings {
            let length = (s.len() as u32).to_le_bytes();
            buffer.write_all(&length).unwrap();
            buffer.write_all(s.as_bytes()).unwrap();
        }

        let got = buffer.strings().unwrap();

        assert_eq!(got, strings);
    }
}
