use crate::guest::ElementType;
use bytemuck::{AnyBitPattern, NoUninit};

/// A primitive value that can be stored directly in a [`crate::guest::Tensor`].
pub trait PrimitiveTensorElement: AnyBitPattern + NoUninit {
    const ELEMENT_TYPE: ElementType;
}

impl PrimitiveTensorElement for u8 {
    const ELEMENT_TYPE: ElementType = ElementType::U8;
}
impl PrimitiveTensorElement for i8 {
    const ELEMENT_TYPE: ElementType = ElementType::I8;
}
impl PrimitiveTensorElement for u16 {
    const ELEMENT_TYPE: ElementType = ElementType::U16;
}
impl PrimitiveTensorElement for i16 {
    const ELEMENT_TYPE: ElementType = ElementType::I16;
}
impl PrimitiveTensorElement for u32 {
    const ELEMENT_TYPE: ElementType = ElementType::U32;
}
impl PrimitiveTensorElement for i32 {
    const ELEMENT_TYPE: ElementType = ElementType::I32;
}
impl PrimitiveTensorElement for f32 {
    const ELEMENT_TYPE: ElementType = ElementType::F32;
}
impl PrimitiveTensorElement for u64 {
    const ELEMENT_TYPE: ElementType = ElementType::U64;
}
impl PrimitiveTensorElement for i64 {
    const ELEMENT_TYPE: ElementType = ElementType::I64;
}
impl PrimitiveTensorElement for f64 {
    const ELEMENT_TYPE: ElementType = ElementType::F64;
}
