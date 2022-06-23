use std::{fmt::Display, str::FromStr};

pub mod runtime_v1 {
    // Note: this also generates a `runtime_v1` module, but it's private and
    // can't be exported. As a workaround, we've wrapped it in another
    // runtime_v1 module and re-exported its contents.
    wit_bindgen_rust::import!("../wit-files/rune/runtime-v1.wit");

    use crate::bindings::ContextExt;

    pub use self::runtime_v1::*;

    use std::{
        fmt::{self, Display, Formatter},
        str::FromStr,
        sync::Once,
    };

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct InvalidElementType {
        pub actual: String,
    }

    impl std::error::Error for InvalidElementType {}

    impl Display for InvalidElementType {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "\"{}\" is not a valid input type, expected one of {:?}",
                self.actual,
                crate::common::element_type::ALL
            )
        }
    }

    impl ElementType {
        pub const ALL: &'static [&'static str] = &[
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
            "utf8",
        ];
        pub const DESCRIPTION: &'static str = "The output type.";
        pub const NAME: &'static str = "element_type";
        pub const NUMERIC: &'static [&'static str] = &[
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        ];

        fn human_name(self) -> &'static str {
            match self {
                ElementType::U8 => "u8",
                ElementType::I8 => "i8",
                ElementType::U16 => "u16",
                ElementType::I16 => "i16",
                ElementType::U32 => "u32",
                ElementType::I32 => "i32",
                ElementType::F32 => "f32",
                ElementType::U64 => "u64",
                ElementType::I64 => "i64",
                ElementType::F64 => "f64",
                ElementType::Utf8 => "utf8",
            }
        }
    }
    impl Display for ElementType {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.human_name().fmt(f)
        }
    }

    impl FromStr for ElementType {
        type Err = InvalidElementType;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "u8" => Ok(ElementType::U8),
                "i8" => Ok(ElementType::I8),
                "u16" => Ok(ElementType::U16),
                "i16" => Ok(ElementType::I16),
                "u32" => Ok(ElementType::U32),
                "i32" => Ok(ElementType::I32),
                "f32" => Ok(ElementType::F32),
                "u64" => Ok(ElementType::U64),
                "i64" => Ok(ElementType::I64),
                "f64" => Ok(ElementType::F64),
                "utf8" => Ok(ElementType::Utf8),
                other => Err(InvalidElementType {
                    actual: other.to_string(),
                }),
            }
        }
    }

    impl ArgumentMetadata {
        /// Register an `element_type` argument which accepts any
        /// [`ElementType`] and defaults to [`ElementType::F32`].
        pub fn element_type() -> Self {
            let element_type = ArgumentMetadata::new(ElementType::NAME);
            element_type.set_description(ElementType::DESCRIPTION);
            element_type.set_default_value(ElementType::F32.human_name());
            element_type.add_hint(&runtime_v1::interpret_as_string_in_enum(
                ElementType::ALL,
            ));
            element_type
        }

        /// Register an `element_type` argument which accepts any **numeric**
        /// [`ElementType`] and defaults to [`ElementType::F32`].
        pub fn numeric_element_type() -> Self {
            let element_type = ArgumentMetadata::new("element_type");
            element_type.set_description(ElementType::DESCRIPTION);
            element_type.set_default_value(ElementType::F32.human_name());
            element_type.add_hint(&runtime_v1::interpret_as_string_in_enum(
                ElementType::NUMERIC,
            ));
            element_type
        }
    }

    impl ContextExt for GraphContext {
        fn _get_argument(&self, name: &str) -> Option<String> {
            self.get_argument(name)
        }
    }

    impl ContextExt for KernelContext {
        fn _get_argument(&self, name: &str) -> Option<String> {
            self.get_argument(name)
        }
    }

    /// Make sure all once-off initialization is done.
    ///
    /// This does things like setting a panic handler which will make sure
    /// panics get logged by the host instead of disappearing.
    pub fn ensure_initialized() {
        static ONCE: Once = Once::new();

        ONCE.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                let msg = info.to_string();
                let location = info.location();
                let meta = LogMetadata {
                    level: LogLevel::Error,
                    file: location.map(|loc| loc.file()),
                    line: location.map(|loc| loc.line()),
                    module: Some(module_path!()),
                    target: module_path!(),
                    name: env!("CARGO_PKG_NAME"),
                };

                log(meta, &msg, &[]);
            }));
        });
    }
}

pub trait ContextErrorExt {
    type InvalidArgument: InvalidArgumentExt;

    fn invalid_argument(inner: Self::InvalidArgument) -> Self;
}

pub trait InvalidArgumentExt {
    fn other(name: &str, msg: impl Display) -> Self;
    fn invalid_value(name: &str, error: impl Display) -> Self;
    fn not_found(name: &str) -> Self;
}

pub trait ContextExt {
    fn _get_argument(&self, name: &str) -> Option<String>;

    fn required_argument<E>(&self, name: &str) -> Result<String, E>
    where
        E: ContextErrorExt,
    {
        self._get_argument(name).ok_or_else(|| {
            E::invalid_argument(E::InvalidArgument::not_found(name))
        })
    }

    fn parse_argument<T, E>(&self, name: &str) -> Result<T, E>
    where
        T: FromStr,
        T::Err: Display,
        E: ContextErrorExt,
    {
        self.required_argument(name)?
            .parse()
            .map_err(|e| E::InvalidArgument::invalid_value(name, e))
            .map_err(E::invalid_argument)
    }

    fn parse_argument_with_default<T, E>(
        &self,
        name: &str,
        default: T,
    ) -> Result<T, E>
    where
        T: FromStr,
        T::Err: Display,
        E: ContextErrorExt,
    {
        let arg = match self._get_argument(name) {
            Some(a) => a,
            None => return Ok(default),
        };

        arg.parse()
            .map_err(|e| E::InvalidArgument::invalid_value(name, e))
            .map_err(E::invalid_argument)
    }
}
