use std::str::FromStr;

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

impl ElementType {
    pub const NAMES: &'static [&'static str] = &[
        "u8",
        "i8",
        "u16",
        "i16",
        "u32",
        "i32",
        "f32",
        "u64",
        "i64",
        "f64",
        "complex64",
        "complex128",
        "utf8",
    ];
}

impl TryFrom<&'_ str> for ElementType {
    type Error = UnknownElementType;

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        match value {
            "u8" | "U8" => Ok(ElementType::U8),
            "i8" | "I8" => Ok(ElementType::I8),
            "u16" | "U16" => Ok(ElementType::U16),
            "i16" | "I16" => Ok(ElementType::I16),
            "u32" | "U32" => Ok(ElementType::U32),
            "i32" | "I32" => Ok(ElementType::I32),
            "f32" | "F32" => Ok(ElementType::F32),
            "u64" | "U64" => Ok(ElementType::U64),
            "i64" | "I64" => Ok(ElementType::I64),
            "f64" | "F64" => Ok(ElementType::F64),
            "complex64" => Ok(ElementType::Complex64),
            "complex128" => Ok(ElementType::Complex128),
            "utf8" | "UTF8" => Ok(ElementType::Utf8),
            other => Err(UnknownElementType(other.to_string())),
        }
    }
}

impl FromStr for ElementType {
    type Err = UnknownElementType;

    fn from_str(s: &str) -> Result<Self, Self::Err> { s.try_into() }
}

impl TryFrom<String> for ElementType {
    type Error = UnknownElementType;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.as_str().try_into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
#[error("Unknown element type, \"{_0}\"")]
pub struct UnknownElementType(String);
