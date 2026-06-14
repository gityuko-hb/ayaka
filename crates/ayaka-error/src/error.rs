//! Rust domain error type and helper constructors.

use alloc::string::String;

use crate::StatusCode;

/// Domain classification for an Ayaka error.
///
/// `ErrorKind` preserves context about which subsystem produced an error so
/// that callers can react appropriately without parsing message strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ErrorKind {
    Shape,
    DType,
    Device,
    Cuda,
    KernelLaunch,
    KVCache,
    Memory,
    Scheduler,
    Request,
    Tokenizer,
    Hub,
    Io,
    InvalidArgument,
    NotFound,
    Unsupported,
    OutOfMemory,
    ContextLengthExceeded,
    AbiMismatch,
    Cancelled,
    Internal,
}

impl ErrorKind {
    /// Returns a stable snake_case name for the kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Shape => "shape",
            ErrorKind::DType => "dtype",
            ErrorKind::Device => "device",
            ErrorKind::Cuda => "cuda",
            ErrorKind::KernelLaunch => "kernel_launch",
            ErrorKind::KVCache => "kv_cache",
            ErrorKind::Memory => "memory",
            ErrorKind::Scheduler => "scheduler",
            ErrorKind::Request => "request",
            ErrorKind::Tokenizer => "tokenizer",
            ErrorKind::Hub => "hub",
            ErrorKind::Io => "io",
            ErrorKind::InvalidArgument => "invalid_argument",
            ErrorKind::NotFound => "not_found",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::OutOfMemory => "out_of_memory",
            ErrorKind::ContextLengthExceeded => "context_length_exceeded",
            ErrorKind::AbiMismatch => "abi_mismatch",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Internal => "internal",
        }
    }

    /// Maps the kind to the closest stable `StatusCode`.
    pub const fn status_code(self) -> StatusCode {
        match self {
            ErrorKind::InvalidArgument => StatusCode::InvalidArgument,
            ErrorKind::Shape => StatusCode::ShapeError,
            ErrorKind::DType => StatusCode::DTypeError,
            ErrorKind::Device | ErrorKind::Cuda => StatusCode::DeviceError,
            ErrorKind::KernelLaunch => StatusCode::KernelLaunchError,
            ErrorKind::OutOfMemory | ErrorKind::Memory => StatusCode::OutOfMemory,
            ErrorKind::ContextLengthExceeded => StatusCode::ContextLengthExceeded,
            ErrorKind::NotFound => StatusCode::NotFound,
            ErrorKind::AbiMismatch => StatusCode::AbiMismatch,
            ErrorKind::Cancelled => StatusCode::Cancelled,
            ErrorKind::Unsupported
            | ErrorKind::KVCache
            | ErrorKind::Scheduler
            | ErrorKind::Request
            | ErrorKind::Tokenizer
            | ErrorKind::Hub
            | ErrorKind::Io
            | ErrorKind::Internal => StatusCode::InternalError,
        }
    }
}

/// The shared error type for the Ayaka inference engine.
///
/// `AyakaError` carries a domain `ErrorKind`, a structured message, and an
/// optional source error. It is designed to be cheaply cloned when no source
/// error is present and to convert cleanly into the C ABI `AyakaStatus`.
#[derive(Debug, Clone)]
pub struct AyakaError {
    kind: ErrorKind,
    message: String,
    #[cfg(feature = "std")]
    source: Option<alloc::sync::Arc<dyn std::error::Error + Send + Sync + 'static>>,
}

/// Alias for fallible Ayaka operations.
pub type Result<T> = core::result::Result<T, AyakaError>;

impl AyakaError {
    /// Creates a new error with the given kind and message.
    pub fn new(
        kind: ErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Creates an error with a source error chained underneath.
    #[cfg(feature = "std")]
    pub fn with_source(
        kind: ErrorKind,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(alloc::sync::Arc::new(source)),
        }
    }

    /// Returns the error's domain kind.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the stable `StatusCode` for this error.
    pub fn status_code(&self) -> StatusCode {
        self.kind.status_code()
    }
}

// Convenience constructors for common error kinds.

impl AyakaError {
    /// `InvalidArgument` error.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidArgument, message)
    }

    /// `Shape` error.
    pub fn shape(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Shape, message)
    }

    /// `DType` error.
    pub fn dtype(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::DType, message)
    }

    /// `Device` error.
    pub fn device(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Device, message)
    }

    /// `Cuda` error.
    pub fn cuda(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Cuda, message)
    }

    /// `OutOfMemory` error.
    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::OutOfMemory, message)
    }

    /// `NotFound` error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    /// `Unsupported` error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    /// `Internal` error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for AyakaError {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AyakaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use std::error::Error;

    #[test]
    fn error_kind_status_code_mapping() {
        assert_eq!(ErrorKind::Shape.status_code(), StatusCode::ShapeError);
        assert_eq!(
            ErrorKind::OutOfMemory.status_code(),
            StatusCode::OutOfMemory
        );
        assert_eq!(
            ErrorKind::ContextLengthExceeded.status_code(),
            StatusCode::ContextLengthExceeded
        );
        assert_eq!(
            ErrorKind::Scheduler.status_code(),
            StatusCode::InternalError
        );
    }

    #[test]
    fn ayaka_error_convenience_constructors() {
        let err = AyakaError::shape("rank mismatch");
        assert_eq!(err.kind(), ErrorKind::Shape);
        assert_eq!(err.status_code(), StatusCode::ShapeError);
        assert!(err.message().contains("rank mismatch"));
    }

    #[test]
    fn ayaka_error_display_contains_kind_and_message() {
        let err = AyakaError::invalid_argument("block_size must be > 0");
        let text = err.to_string();
        assert!(text.contains("invalid_argument"));
        assert!(text.contains("block_size"));
    }

    #[test]
    fn ayaka_error_with_source() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err = AyakaError::with_source(ErrorKind::Io, "read failed", inner);
        assert!(err.source().is_some());
        let text = err.to_string();
        assert!(text.contains("read failed"));
    }
}
