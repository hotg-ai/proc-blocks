/// A builder for serializing a string tensor.
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
    pub const fn new() -> Self { StringBuilder::with_buffer(Vec::new()) }

    pub const fn with_buffer(buffer: Vec<u8>) -> Self {
        StringBuilder { buffer }
    }

    pub fn finish(self) -> Vec<u8> { self.buffer }

    pub fn buffer(&self) -> &[u8] { &self.buffer }

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
