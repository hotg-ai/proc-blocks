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
