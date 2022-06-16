pub mod runtime_v2 {
    wit_bindgen_wasmer::export!("../wit-files/rune/runtime-v2.wit");
    #[doc(inline)]
    pub use self::runtime_v2::*;
}

pub mod proc_block_v2 {
    use std::{error::Error, fmt::Display, num::NonZeroU32};

    #[doc(inline)]
    pub use proc_block_v2::*;
    pub use TensorResult as Tensor;

    use serde::ser::{Serialize, SerializeSeq, SerializeStruct, Serializer};

    wit_bindgen_wasmer::import!("../wit-files/rune/proc-block-v2.wit");

    impl Serialize for Metadata {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let Metadata {
                name,
                version,
                description,
                repository,
                homepage,
                tags,
                arguments,
                inputs,
                outputs,
            } = self;

            let mut ser = serializer.serialize_struct("Metadata", 8)?;

            ser.serialize_field("name", name)?;
            ser.serialize_field("version", version)?;
            ser.serialize_field("description", description)?;
            ser.serialize_field("repository", repository)?;
            ser.serialize_field("homepage", homepage)?;
            ser.serialize_field("tags", tags)?;
            ser.serialize_field("arguments", arguments)?;
            ser.serialize_field("inputs", inputs)?;
            ser.serialize_field("outputs", outputs)?;

            ser.end()
        }
    }

    impl Serialize for TensorMetadata {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let TensorMetadata {
                name,
                description,
                hints,
            } = self;
            let mut ser = serializer.serialize_struct("TensorMetadata", 2)?;

            ser.serialize_field("name", name)?;
            ser.serialize_field("description", description)?;
            ser.serialize_field("hints", hints)?;

            ser.end()
        }
    }

    impl Serialize for ArgumentMetadata {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let ArgumentMetadata {
                name,
                description,
                hints,
                default_value,
            } = self;
            let mut ser = serializer.serialize_struct("ArgumentMetadata", 2)?;

            ser.serialize_field("name", name)?;
            ser.serialize_field("description", description)?;
            ser.serialize_field("hints", hints)?;
            ser.serialize_field("default_value", default_value)?;

            ser.end()
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

    impl Serialize for TensorHint {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            #[derive(serde::Serialize)]
            enum TensorHint<'a> {
                Other(&'a str),
                MediaType(MediaType),
            }

            match self {
                Self::Other(other) => TensorHint::Other(other).serialize(ser),
                Self::MediaType(ty) => TensorHint::MediaType(*ty).serialize(ser),
            }
        }
    }

    impl Serialize for ArgumentHint {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            #[derive(serde::Serialize)]
            enum ArgumentHint<'a> {
                Between((&'a str, &'a str)),
                OneOf(&'a [String]),
                NonNegativeNumber,
                ArgumentType(ArgumentType),
            }

            let arg = match self {
                Self::Between((low, high)) => {
                    ArgumentHint::Between((low, high))
                },
                Self::OneOf(x) => ArgumentHint::OneOf(x),
                Self::NonNegativeNumber => ArgumentHint::NonNegativeNumber,
                Self::ArgumentType(ty) => ArgumentHint::ArgumentType(*ty),
            };

            arg.serialize(ser)
        }
    }

    impl Serialize for ArgumentType {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            format!("{self:?}").serialize(ser)
        }
    }

    impl Serialize for MediaType {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            format!("{self:?}").serialize(ser)
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

    impl Display for CreateError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                CreateError::Argument(_) => write!(f, "Unable to create the node because of an issue with the arguments"),
                CreateError::Other(msg) => write!(f, "{msg}"),
            }
        }
    }

    impl Error for CreateError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                CreateError::Argument(a) => Some(a),
                CreateError::Other(_) => None,
            }
        }
    }

    impl Display for ArgumentError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let name = &self.name;
            write!(f, "The \"{name}\" argument was invalid")
        }
    }

    impl Error for ArgumentError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.reason)
        }
    }

    impl Display for ArgumentErrorReason {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ArgumentErrorReason::Other(msg) => write!(f, "{msg}"),
                ArgumentErrorReason::NotFound => {
                    write!(f, "The argument wasn't provided")
                },
                ArgumentErrorReason::InvalidValue(reason) => {
                    write!(f, "Invalid value: {reason}")
                },
                ArgumentErrorReason::ParseFailed(reason) => {
                    write!(f, "Parse failed: {reason}")
                },
            }
        }
    }

    impl Error for ArgumentErrorReason {}
}
