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
