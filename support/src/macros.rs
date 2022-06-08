#[macro_export]
macro_rules! generate_support {
    ($($proc_block:ident)::*) => {
        mod support {
            use std::{fmt::{self, Display, Formatter}, str::FromStr};
            use $($proc_block)::*::*;

            pub fn parse_arg<T>(args: &[Argument], name: &str) -> Result<T, ArgumentError>
            where
                T: FromStr,
                T::Err: Display,
            {
                for arg in args {
                    if arg.name == name {
                        return arg.value.parse::<T>().map_err(|e| ArgumentError {
                            name: name.to_string(),
                            reason: ArgumentErrorReason::InvalidValue(e.to_string()),
                        });
                    }
                }

                Err(ArgumentError {
                    name: name.to_string(),
                    reason: ArgumentErrorReason::NotFound,
                })
            }

            pub fn get_input_tensor<'t>(tensors: &'t [Tensor], name: &str) -> Result<&'t Tensor, KernelError> {
                tensors.iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| KernelError::InvalidInput(InvalidInput {
                            name: name.to_string(),
                            reason: InvalidInputReason::NotFound,
                        }))
            }

            impl Tensor {
                pub fn view<T>(&self) -> Result<$crate::ndarray::ArrayViewD<'_, T>, KernelError>
                where
                    T: ValueType,
                {
                    if self.element_type != T::ELEMENT_TYPE {
                        return Err(KernelError::InvalidInput(InvalidInput {
                            name: self.name.clone(),
                            reason: InvalidInputReason::UnsupportedShape,
                        }));
                    }

                    todo!();
                }

                pub fn view_1d<T>(&self) -> Result<$crate::ndarray::ArrayView1<'_, T>, KernelError>
                where
                    T: ValueType,
                {
                    let array = self.view::<T>()?;

                    array.into_dimensionality()
                        .map_err(|e| KernelError::InvalidInput(InvalidInput {
                            name: self.name.clone(),
                            reason: InvalidInputReason::InvalidValue(format!("Incorrect dimensions: {e}")),
                        }))
                }
            }

            pub trait ValueType: $crate::ValueType {
                const ELEMENT_TYPE: ElementType;
            }

            impl ValueType for u8 { const ELEMENT_TYPE: ElementType = ElementType::U8; }
            impl ValueType for i8 { const ELEMENT_TYPE: ElementType = ElementType::I8; }
            impl ValueType for u16 { const ELEMENT_TYPE: ElementType = ElementType::U16; }
            impl ValueType for i16 { const ELEMENT_TYPE: ElementType = ElementType::I16; }
            impl ValueType for u32 { const ELEMENT_TYPE: ElementType = ElementType::U32; }
            impl ValueType for i32 { const ELEMENT_TYPE: ElementType = ElementType::I32; }
            impl ValueType for f32 { const ELEMENT_TYPE: ElementType = ElementType::F32; }
            impl ValueType for u64 { const ELEMENT_TYPE: ElementType = ElementType::U64; }
            impl ValueType for i64 { const ELEMENT_TYPE: ElementType = ElementType::I64; }
            impl ValueType for f64 { const ELEMENT_TYPE: ElementType = ElementType::F64; }

            impl Display for KernelError {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    match self {
                        KernelError::InvalidInput(i) => i.fmt(f),
                        KernelError::Other(msg) => write!(f, "{}", msg),
                    }
                }
            }

            impl std::error::Error for KernelError {
                fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                    match self {
                        KernelError::InvalidInput(i) => todo!(),
                        KernelError::Other(_) => None,
                    }
                }
            }

            impl Display for InvalidInput {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    write!(f, "The \"{}\" input tensor was invalid", self.name)
                }
            }

            impl Display for InvalidInputReason {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    match self {
                        InvalidInputReason::Other(msg) => write!(f, "{msg}"),
                        InvalidInputReason::NotFound => write!(f, "Not found"),
                        InvalidInputReason::InvalidValue(msg) => write!(f, "Invalid value: {msg}"),
                        InvalidInputReason::UnsupportedShape => write!(f, "Unsupported shape"),
                    }
                }
            }

            impl std::error::Error for InvalidInputReason {}

            impl Display for ArgumentError {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    write!(f, "The \"{}\" argument is invalid", self.name)
                }
            }

            impl std::error::Error for ArgumentError {
                fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                    Some(&self.reason)
                }
            }

            impl Display for ArgumentErrorReason {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    match self {
                        ArgumentErrorReason::Other(msg) => write!(f, "{msg}"),
                        ArgumentErrorReason::NotFound => {
                            write!(f, "The argument wasn't defined")
                        },
                        ArgumentErrorReason::InvalidValue(msg) => {
                            write!(f, "Invalid value: {msg}")
                        },
                    }
                }
            }

            impl std::error::Error for ArgumentErrorReason {}
        }
    };
}
