//! Context helpers for `Result` and `Option`.

use alloc::string::String;

use crate::{AyakaError, ErrorKind, Result};

/// Extension trait for attaching context to `Result<T, E>`.
pub trait ResultExt<T, E> {
    /// Maps an error into an `AyakaError` with the given kind and message.
    fn ayaka_context(self, kind: ErrorKind, message: impl Into<String>) -> Result<T>;

    /// Maps an error into an `AyakaError` with a lazily computed message.
    fn ayaka_with_context<F: FnOnce(&E) -> String>(self, kind: ErrorKind, f: F) -> Result<T>;
}

impl<T, E: core::fmt::Display> ResultExt<T, E> for core::result::Result<T, E> {
    fn ayaka_context(self, kind: ErrorKind, message: impl Into<String>) -> Result<T> {
        self.map_err(|e| AyakaError::new(kind, alloc::format!("{}: {}", message.into(), e)))
    }

    fn ayaka_with_context<F: FnOnce(&E) -> String>(self, kind: ErrorKind, f: F) -> Result<T> {
        self.map_err(|e| AyakaError::new(kind, f(&e)))
    }
}

/// Extension trait for converting `Option<T>` into `Result<T>`.
pub trait OptionExt<T> {
    /// Converts `None` into an `AyakaError` with the given kind and message.
    fn ok_or_ayaka(self, kind: ErrorKind, message: impl Into<String>) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_ayaka(self, kind: ErrorKind, message: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| AyakaError::new(kind, message))
    }
}
