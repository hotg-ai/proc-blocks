use ndarray::{ArrayViewD, ArrayViewMutD, ShapeError};

/// Extension traits added to a byte buffer.
pub trait BufferExt {
    /// Reinterpret this byte buffer as a slice of `T`'s.
    fn elements<T: ValueType>(&self) -> &[T];
    /// Reinterpret this byte buffer as a mutable slice of `T`'s.
    fn elements_mut<T: ValueType>(&mut self) -> &mut [T];

    /// View the buffer as a multi-dimensional array.
    fn view<T: ValueType>(
        &self,
        dimensions: &[u32],
    ) -> Result<ArrayViewD<'_, T>, ShapeError> {
        let elements = self.elements();
        let dimensions: Vec<_> = dimensions
            .iter()
            .map(|&dim| {
                usize::try_from(dim).expect("Conversion should never fail")
            })
            .collect();
        ArrayViewD::from_shape(dimensions, elements)
    }

    /// View the buffer as a mutable multi-dimensional array.
    fn view_mut<T: ValueType>(
        &mut self,
        dimensions: &[u32],
    ) -> Result<ArrayViewMutD<'_, T>, ShapeError> {
        let elements = self.elements_mut();
        let dimensions: Vec<_> = dimensions
            .iter()
            .map(|&dim| {
                usize::try_from(dim).expect("Conversion should never fail")
            })
            .collect();
        ArrayViewMutD::from_shape(dimensions, elements)
    }
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
}

/// A value type which can be reinterpreted to/from a byte buffer.
///
/// # Safety
///
/// It must be safe to reinterpret a `&[u8]` as a `&[Self]`. Among other things,
/// this means:
/// - The type must not have any padding (observing padding bytes is UB)
/// - It must not contain any fields with indirection (e.g. we don't want to
///   interpret random bytes as a pointer... that's just asking for trouble)
pub unsafe trait ValueType: Sized {}

unsafe impl ValueType for u8 {}
unsafe impl ValueType for i8 {}
unsafe impl ValueType for u16 {}
unsafe impl ValueType for i16 {}
unsafe impl ValueType for u32 {}
unsafe impl ValueType for i32 {}
unsafe impl ValueType for f32 {}
unsafe impl ValueType for u64 {}
unsafe impl ValueType for i64 {}
unsafe impl ValueType for f64 {}

#[cfg(test)]
mod tests {
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
}
