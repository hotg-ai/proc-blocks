//! Common arguments that are used across proc-blocks.

pub mod element_type {
    pub const NAME: &str = "element_type";
    pub const DESCRIPTION: &str = "The output type.";
    pub const ALL: &[&str] = &[
        "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
        "utf8",
    ];
    pub const NUMERIC: &[&str] = &[
        "u8", "i8", "u16", "i16", "u32", "i32", "f32", "u64", "i64", "f64",
    ];
}
