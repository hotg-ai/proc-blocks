use std::fmt::{self, Debug, Formatter};

use ndarray::{ErrorKind, ShapeError};

/// A builder for serializing multiple UTF-8 strings to a flat byte array.
///
/// #Examples
///
/// ```rust
/// # use hotg_rune_proc_blocks::StringBuilder;
/// // Construct a new string builder and add some strings to it
/// let mut builder = StringBuilder::new();
/// builder.push("this").push("is").push("a").push("sentence");
///
/// // once all the strings have been added, we can get the serialized tensor.
/// let buffer: Vec<u8> = builder.finish();
///
/// let strings: Vec<&str> = hotg_rune_proc_blocks::decode_strings(&buffer)?;
///
/// assert_eq!(strings, &["this", "is", "a", "sentence"]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(PartialEq, Eq)]
pub struct StringBuilder {
    buffer: Vec<u8>,
}

impl StringBuilder {
    /// Construct a [`StringBuilder`] with an empty string.
    pub const fn new() -> Self { StringBuilder::with_buffer(Vec::new()) }

    /// Construct a [`StringBuilder`] using an existing buffer, allowing the
    /// caller to reuse allocations.
    ///
    /// Note that this *doesn't* clear the buffer before it starts writing
    /// strings to it.
    pub const fn with_buffer(buffer: Vec<u8>) -> Self {
        StringBuilder { buffer }
    }

    /// Consume the [`StringBuilder`], returning the buffer.
    pub fn finish(self) -> Vec<u8> { self.buffer }

    /// Get a readonly reference to the serialized bytes.
    pub fn buffer(&self) -> &[u8] { &self.buffer }

    /// Add a string to the buffer.
    pub fn push(&mut self, string: &str) -> &mut Self {
        let length = u32::try_from(string.len())
            .expect("The string length doesn't fit in a u32");
        let length = length.to_le_bytes();
        self.buffer.extend(&length);
        self.buffer.extend(string.as_bytes());

        self
    }
}

impl Debug for StringBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StringBuilder").finish_non_exhaustive()
    }
}

impl Default for StringBuilder {
    fn default() -> Self { StringBuilder::new() }
}

/// Decode list of strings from their serialized form.
///
/// See [`StringBuilder`] for how to serialize a list of strings.
pub fn decode_strings(raw: &[u8]) -> Result<Vec<&str>, ShapeError> {
    const HEADER_SIZE: usize = std::mem::size_of::<u32>();

    let mut strings = Vec::new();
    let mut buffer = raw;

    while !buffer.is_empty() {
        if buffer.len() < HEADER_SIZE {
            // We don't have enough bytes remaining for a full length field,
            // so something is probably wrong with our buffer.
            return Err(ShapeError::from_kind(ErrorKind::OutOfBounds));
        }

        let (len, rest) = buffer.split_at(HEADER_SIZE);

        let len: [u8; HEADER_SIZE] = len.try_into().expect("Unreachable");
        let len = u32::from_le_bytes(len);
        let len = usize::try_from(len).expect("Unreachable");

        if rest.len() < len {
            // We don't have enough bytes left in the buffer to read a
            // string with this length.
            return Err(ShapeError::from_kind(ErrorKind::OutOfBounds));
        }

        let (s, rest) = rest.split_at(len);

        match std::str::from_utf8(s) {
            Ok(s) => strings.push(s),
            Err(_) => {
                // The string wasn't valid UTF-8. We're probably using the
                // wrong ShapeError here, but our alternative would be
                // introducing our own error type and that seems overkill.
                return Err(ShapeError::from_kind(
                    ErrorKind::IncompatibleLayout,
                ));
            },
        }

        buffer = rest;
    }

    Ok(strings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_some_strings() {
        let mut builder = StringBuilder::new();
        builder.push("this").push("is").push("a").push("sentence");
        let buffer = builder.finish();

        let strings = decode_strings(&buffer).unwrap();

        assert_eq!(strings, &["this", "is", "a", "sentence"]);
    }
}
