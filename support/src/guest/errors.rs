use std::fmt::{self, Display, Formatter};

use crate::guest::bindings::*;

impl KernelError {
    pub fn other(reason: impl Display) -> Self {
        KernelError::Other(reason.to_string())
    }

    pub fn unsupported_shape(tensor_name: impl Into<String>) -> Self {
        KernelError::InvalidInput(InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::UnsupportedShape,
        })
    }
}

impl PartialEq for KernelError {
    fn eq(&self, other: &KernelError) -> bool {
        match (self, other) {
            (KernelError::Other(left), KernelError::Other(right)) => {
                left == right
            },
            (
                KernelError::InvalidInput(left),
                KernelError::InvalidInput(right),
            ) => left == right,
            (KernelError::Other(_), _) | (KernelError::InvalidInput(_), _) => {
                false
            },
        }
    }
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
            KernelError::InvalidInput(i) => Some(i),
            KernelError::Other(_) => None,
        }
    }
}

impl From<InvalidInput> for KernelError {
    fn from(e: InvalidInput) -> Self { KernelError::InvalidInput(e) }
}

impl CreateError {
    pub fn other(error: impl Display) -> Self {
        CreateError::Other(error.to_string())
    }
}

impl From<ArgumentError> for CreateError {
    fn from(e: ArgumentError) -> Self { CreateError::Argument(e) }
}

impl InvalidInput {
    pub fn unsupported_shape(tensor_name: impl Into<String>) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::UnsupportedShape,
        }
    }

    pub fn not_found(tensor_name: impl Into<String>) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::NotFound,
        }
    }

    pub fn invalid_value(
        tensor_name: impl Into<String>,
        error: impl Display,
    ) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::InvalidValue(error.to_string()),
        }
    }

    pub fn other(tensor_name: impl Into<String>, reason: impl Display) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::Other(reason.to_string()),
        }
    }
}

impl PartialEq for InvalidInput {
    fn eq(&self, other: &InvalidInput) -> bool {
        let InvalidInput { name, reason } = self;

        name == &other.name && reason == &other.reason
    }
}

impl std::error::Error for InvalidInput {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.reason)
    }
}

impl Display for InvalidInput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "The \"{}\" input tensor was invalid", self.name)
    }
}

impl PartialEq for InvalidInputReason {
    fn eq(&self, other: &InvalidInputReason) -> bool {
        match (self, other) {
            (
                InvalidInputReason::Other(left),
                InvalidInputReason::Other(right),
            ) => left == right,
            (
                InvalidInputReason::InvalidValue(left),
                InvalidInputReason::InvalidValue(right),
            ) => left == right,
            (InvalidInputReason::NotFound, InvalidInputReason::NotFound) => {
                true
            },
            (
                InvalidInputReason::UnsupportedShape,
                InvalidInputReason::UnsupportedShape,
            ) => true,
            (InvalidInputReason::Other(_), _)
            | (InvalidInputReason::InvalidValue(_), _)
            | (InvalidInputReason::NotFound, _)
            | (InvalidInputReason::UnsupportedShape, _) => false,
        }
    }
}

impl Display for InvalidInputReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            InvalidInputReason::Other(msg) => write!(f, "{msg}"),
            InvalidInputReason::NotFound => write!(f, "Not found"),
            InvalidInputReason::InvalidValue(msg) => {
                write!(f, "Invalid value: {msg}")
            },
            InvalidInputReason::UnsupportedShape => {
                write!(f, "Unsupported shape")
            },
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
