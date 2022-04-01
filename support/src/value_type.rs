/// Extension traits added to a byte buffer.
pub trait BufferExt {
    /// Reinterpret this byte buffer as a slice of `T`'s.
    fn elements<T: ValueType>(&self) -> &[T];
    /// Reinterpret this byte buffer as a mutable slice of `T`'s.
    fn elements_mut<T: ValueType>(&mut self) -> &mut [T];
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
