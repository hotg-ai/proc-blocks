use std::{
    convert::Infallible,
    fmt::{self, Display, Formatter},
};

use crate::guest::bindings::*;

impl RunError {
    pub fn other(reason: impl Display) -> Self {
        RunError::Other(reason.to_string())
    }

    pub fn missing_input(name: impl Into<String>) -> Self {
        RunError::InvalidInput(InvalidInput::not_found(name))
    }
}

impl PartialEq for RunError {
    fn eq(&self, other: &RunError) -> bool {
        match (self, other) {
            (RunError::Other(left), RunError::Other(right)) => left == right,
            (RunError::InvalidInput(left), RunError::InvalidInput(right)) => {
                left == right
            },
            (RunError::Other(_), _) | (RunError::InvalidInput(_), _) => false,
        }
    }
}

impl Display for RunError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RunError::InvalidInput(i) => i.fmt(f),
            RunError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunError::InvalidInput(i) => Some(i),
            RunError::Other(_) => None,
        }
    }
}

impl From<InvalidInput> for RunError {
    fn from(e: InvalidInput) -> Self { RunError::InvalidInput(e) }
}

impl CreateError {
    pub fn other(error: impl Display) -> Self {
        CreateError::Other(error.to_string())
    }
}

impl From<Infallible> for CreateError {
    fn from(v: Infallible) -> Self { match v {} }
}

impl From<ArgumentError> for CreateError {
    fn from(e: ArgumentError) -> Self { CreateError::Argument(e) }
}

impl InvalidInput {
    pub fn incompatible_dimensions(tensor_name: impl Into<String>) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::IncompatibleDimensions,
        }
    }

    pub fn incompatible_element_type(tensor_name: impl Into<String>) -> Self {
        InvalidInput {
            name: tensor_name.into(),
            reason: InvalidInputReason::IncompatibleElementType,
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
                InvalidInputReason::IncompatibleDimensions,
                InvalidInputReason::IncompatibleDimensions,
            ) => true,
            (
                InvalidInputReason::IncompatibleElementType,
                InvalidInputReason::IncompatibleElementType,
            ) => true,
            (InvalidInputReason::Other(_), _)
            | (InvalidInputReason::InvalidValue(_), _)
            | (InvalidInputReason::NotFound, _)
            | (InvalidInputReason::IncompatibleElementType, _)
            | (InvalidInputReason::IncompatibleDimensions, _) => false,
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
            InvalidInputReason::IncompatibleDimensions => {
                write!(f, "Incompatible dimensions")
            },
            InvalidInputReason::IncompatibleElementType => {
                write!(f, "Incompatible element type")
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
            ArgumentErrorReason::ParseFailed(e) => {
                write!(f, "Parse failed: {e}")
            },
        }
    }
}

impl std::error::Error for ArgumentErrorReason {}
