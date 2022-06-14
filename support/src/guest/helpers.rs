use std::{fmt::Display, str::FromStr};

use crate::guest::{Argument, ArgumentError, ArgumentErrorReason};

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
