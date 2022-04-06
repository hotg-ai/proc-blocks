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

pub trait SliceExt {
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

impl<T: ValueType> SliceExt for [T] {
    fn as_bytes(&self) -> &[u8] {
        let length = std::mem::size_of_val(self);

        unsafe { std::slice::from_raw_parts(self.as_ptr().cast(), length) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        let length = std::mem::size_of_val(self);

        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr().cast(), length)
        }
    }
}
