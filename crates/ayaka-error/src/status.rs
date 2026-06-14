//! Stable status codes used across the C ABI and Rust error surfaces.

/// Stable status code returned across the Ayaka C ABI.
///
/// Values are fixed and must never be renumbered. New codes append to the
/// enum and reserve values 14–31 for future expansion.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StatusCode {
    /// Operation completed successfully.
    Ok = 0,
    /// An argument was invalid or malformed.
    InvalidArgument = 1,
    /// A data-type mismatch or unsupported dtype was encountered.
    DTypeError = 2,
    /// A tensor shape mismatch or invalid shape was encountered.
    ShapeError = 3,
    /// A device-related error (missing device, bad ordinal, etc.).
    DeviceError = 4,
    /// The requested operation is not supported.
    Unsupported = 5,
    /// Allocation failed because the device or host ran out of memory.
    OutOfMemory = 6,
    /// The requested context length exceeds the model or KV limit.
    ContextLengthExceeded = 7,
    /// A requested resource was not found.
    NotFound = 8,
    /// Rust-core and native ABI versions do not match.
    AbiMismatch = 9,
    /// A CUDA / native kernel launch failed.
    KernelLaunchError = 10,
    /// A stream operation failed or an invalid stream was supplied.
    StreamError = 11,
    /// An internal invariant was violated.
    InternalError = 12,
    /// The operation was cancelled.
    Cancelled = 13,
}

impl StatusCode {
    /// Returns `true` if the status represents success.
    #[inline]
    pub const fn is_ok(self) -> bool {
        matches!(self, StatusCode::Ok)
    }

    /// Returns `true` if the status represents an error.
    #[inline]
    pub const fn is_err(self) -> bool {
        !self.is_ok()
    }

    /// Returns the raw `i32` discriminant.
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// Converts a raw `i32` into a `StatusCode`.
    ///
    /// Unknown values are mapped to `StatusCode::InternalError` so that the
    /// ABI never returns an invalid code to callers.
    #[inline]
    pub const fn from_i32(code: i32) -> Self {
        match code {
            0 => StatusCode::Ok,
            1 => StatusCode::InvalidArgument,
            2 => StatusCode::DTypeError,
            3 => StatusCode::ShapeError,
            4 => StatusCode::DeviceError,
            5 => StatusCode::Unsupported,
            6 => StatusCode::OutOfMemory,
            7 => StatusCode::ContextLengthExceeded,
            8 => StatusCode::NotFound,
            9 => StatusCode::AbiMismatch,
            10 => StatusCode::KernelLaunchError,
            11 => StatusCode::StreamError,
            12 => StatusCode::InternalError,
            13 => StatusCode::Cancelled,
            _ => StatusCode::InternalError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_code_discriminants_match_c_abi() {
        assert_eq!(StatusCode::Ok.as_i32(), 0);
        assert_eq!(StatusCode::InvalidArgument.as_i32(), 1);
        assert_eq!(StatusCode::DTypeError.as_i32(), 2);
        assert_eq!(StatusCode::ShapeError.as_i32(), 3);
        assert_eq!(StatusCode::DeviceError.as_i32(), 4);
        assert_eq!(StatusCode::Unsupported.as_i32(), 5);
        assert_eq!(StatusCode::OutOfMemory.as_i32(), 6);
        assert_eq!(StatusCode::ContextLengthExceeded.as_i32(), 7);
        assert_eq!(StatusCode::NotFound.as_i32(), 8);
        assert_eq!(StatusCode::AbiMismatch.as_i32(), 9);
        assert_eq!(StatusCode::KernelLaunchError.as_i32(), 10);
        assert_eq!(StatusCode::StreamError.as_i32(), 11);
        assert_eq!(StatusCode::InternalError.as_i32(), 12);
        assert_eq!(StatusCode::Cancelled.as_i32(), 13);
    }

    #[test]
    fn status_code_from_i32_round_trip() {
        for code in 0..=13 {
            let status = StatusCode::from_i32(code);
            assert_eq!(status.as_i32(), code);
        }
    }

    #[test]
    fn status_code_unknown_maps_to_internal() {
        assert_eq!(StatusCode::from_i32(999), StatusCode::InternalError);
        assert_eq!(StatusCode::from_i32(-1), StatusCode::InternalError);
    }

    #[test]
    fn status_code_ok_err_helpers() {
        assert!(StatusCode::Ok.is_ok());
        assert!(!StatusCode::Ok.is_err());
        assert!(StatusCode::OutOfMemory.is_err());
        assert!(!StatusCode::OutOfMemory.is_ok());
    }
}
