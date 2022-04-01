pub trait BufferExt {
    fn elements<T: ValueType>(&self) -> &[T];
    fn elements_mut<T: ValueType>(&mut self) -> &mut [T];
}

impl BufferExt for [u8] {
    fn elements<T: ValueType>(&self) -> &[T] { unsafe { T::from_bytes(self) } }

    fn elements_mut<T: ValueType>(&mut self) -> &mut [T] {
        unsafe { T::from_bytes_mut(self) }
    }
}

pub unsafe trait ValueType: Sized {
    unsafe fn from_bytes(bytes: &[u8]) -> &[Self];
    unsafe fn from_bytes_mut(bytes: &mut [u8]) -> &mut [Self];
}

macro_rules! impl_value_type {
        ($( $type:ty ),* $(,)?) => {
            $(
                unsafe impl ValueType for $type {
                    unsafe fn from_bytes(bytes: &[u8]) -> &[Self] {
                        let (start, middle, end) = bytes.align_to::<$type>();
                        assert!(start.is_empty());
                        assert!(end.is_empty());
                        middle
                    }

                    unsafe fn from_bytes_mut(bytes: &mut [u8]) -> &mut [Self] {
                        let (start, middle, end) = bytes.align_to_mut::<$type>();
                        assert!(start.is_empty());
                        assert!(end.is_empty());
                        middle
                    }
                }
            )*
        };
    }

impl_value_type!(u8, i8, u16, i16, u32, i32, f32, u64, i64, f64);
