pub mod runtime_v2 {
    wit_bindgen_wasmer::export!("../wit-files/rune/runtime-v2.wit");
    #[doc(inline)]
    pub use self::runtime_v2::*;
}

pub mod proc_block_v2 {
    use std::{
        num::NonZeroU32,
    };

    #[doc(inline)]
    pub use proc_block_v2::*;

    use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};

    wit_bindgen_wasmer::import!("../wit-files/rune/proc-block-v2.wit");

    impl Serialize for Metadata {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            todo!()
        }
    }

    impl Serialize for TensorConstraints {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let TensorConstraints { inputs, outputs } = self;
            let mut ser =
                serializer.serialize_struct("TensorConstraints", 2)?;

            ser.serialize_field("inputs", inputs)?;
            ser.serialize_field("outputs", outputs)?;

            ser.end()
        }
    }

    impl Serialize for TensorConstraint {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let TensorConstraint {
                name,
                element_type,
                dimensions,
            } = self;
            let mut ser = serializer.serialize_struct("TensorConstraint", 3)?;

            ser.serialize_field("name", name)?;
            ser.serialize_field("element_type", element_type)?;
            ser.serialize_field("dimensions", dimensions)?;

            ser.end()
        }
    }

    impl Serialize for ElementTypeConstraint {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut ser = serializer
                .serialize_seq(Some(self.bits().count_ones() as usize))?;

            if self.contains(ElementTypeConstraint::U8) {
                ser.serialize_element(&ElementType::U8)?;
            }
            if self.contains(ElementTypeConstraint::I8) {
                ser.serialize_element(&ElementType::I8)?;
            }
            if self.contains(ElementTypeConstraint::U16) {
                ser.serialize_element(&ElementType::U16)?;
            }
            if self.contains(ElementTypeConstraint::I16) {
                ser.serialize_element(&ElementType::I16)?;
            }
            if self.contains(ElementTypeConstraint::U32) {
                ser.serialize_element(&ElementType::U32)?;
            }
            if self.contains(ElementTypeConstraint::I32) {
                ser.serialize_element(&ElementType::I32)?;
            }
            if self.contains(ElementTypeConstraint::F32) {
                ser.serialize_element(&ElementType::F32)?;
            }
            if self.contains(ElementTypeConstraint::U64) {
                ser.serialize_element(&ElementType::U64)?;
            }
            if self.contains(ElementTypeConstraint::I64) {
                ser.serialize_element(&ElementType::I64)?;
            }
            if self.contains(ElementTypeConstraint::F64) {
                ser.serialize_element(&ElementType::F64)?;
            }
            if self.contains(ElementTypeConstraint::COMPLEX64) {
                ser.serialize_element(&ElementType::Complex64)?;
            }
            if self.contains(ElementTypeConstraint::COMPLEX128) {
                ser.serialize_element(&ElementType::Complex128)?;
            }
            if self.contains(ElementTypeConstraint::UTF8) {
                ser.serialize_element(&ElementType::Utf8)?;
            }

            ser.end()
        }
    }

    impl Serialize for ElementType {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match self {
                ElementType::U8 => "u8".serialize(serializer),
                ElementType::I8 => "i8".serialize(serializer),
                ElementType::U16 => "u16".serialize(serializer),
                ElementType::I16 => "i16".serialize(serializer),
                ElementType::U32 => "u32".serialize(serializer),
                ElementType::I32 => "i32".serialize(serializer),
                ElementType::F32 => "f32".serialize(serializer),
                ElementType::U64 => "u64".serialize(serializer),
                ElementType::I64 => "i64".serialize(serializer),
                ElementType::F64 => "f64".serialize(serializer),
                ElementType::Complex64 => "complex64".serialize(serializer),
                ElementType::Complex128 => "complex128".serialize(serializer),
                ElementType::Utf8 => "utf8".serialize(serializer),
            }
        }
    }

    impl Serialize for Dimensions {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            #[derive(serde::Serialize)]
            enum DimensionsWrapper {
                Dynamic,
                Fixed(Vec<Option<NonZeroU32>>),
            }

            let dim = match self {
                Dimensions::Dynamic => DimensionsWrapper::Dynamic,
                Dimensions::Fixed(dims) => DimensionsWrapper::Fixed(
                    dims.iter().copied().map(NonZeroU32::new).collect(),
                ),
            };

            dim.serialize(serializer)
        }
    }
}
