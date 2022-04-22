/// Serialize a string tensor so it can be passed to the runtime.
///
/// # Examples
///
/// ```
/// use hotg_rune_proc_blocks::{BufferExt, string_tensor_from_ndarray};
///
/// let tensor = ndarray::arr2(&[
///     ["this", "is", "a", "sentence"],
///     ["and", "this", "is", "another"],
/// ]);
///
/// let serialized: Vec<u8> = string_tensor_from_ndarray(&tensor);
///
/// let deserialized = serialized.string_view(&[2, 4]).unwrap();
/// assert_eq!(deserialized, tensor.into_dyn());
/// ```
pub fn string_tensor_from_ndarray<S, Data, Dim>(
    array: &ndarray::ArrayBase<Data, Dim>,
) -> Vec<u8>
where
    Dim: ndarray::Dimension,
    Data: ndarray::Data<Elem = S>,
    S: AsRef<str>,
{
    let mut builder = StringBuilder::new();

    for s in array.iter() {
        builder.push(s.as_ref());
    }

    builder.finish()
}

/// A builder for serializing multiple UTF-8 strings to a flat byte array.
///
/// #Examples
///
/// ```rust
/// # use hotg_rune_proc_blocks::StringBuilder;
/// use hotg_rune_proc_blocks::BufferExt;
/// // Construct a new string builder and add some strings to it
/// let mut builder = StringBuilder::new();
/// builder.push("this").push("is").push("a").push("sentence");
///
/// // once all the strings have been added, we can get the serialized tensor.
/// let buffer: Vec<u8> = builder.finish();
///
/// // The BufferExt trait lets us deserialize the strings again.
/// let strings: Vec<&str> = buffer.strings()?;
///
/// assert_eq!(strings, &["this", "is", "a", "sentence"]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(PartialEq)]
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

impl Default for StringBuilder {
    fn default() -> Self { StringBuilder::new() }
}

#[cfg(test)]
mod tests {
    use crate::BufferExt;

    use super::*;

    #[test]
    fn round_trip_some_strings() {
        let mut builder = StringBuilder::new();
        builder.push("this").push("is").push("a").push("sentence");
        let buffer = builder.finish();

        let strings = buffer.strings().unwrap();

        assert_eq!(strings, &["this", "is", "a", "sentence"]);
    }
}
