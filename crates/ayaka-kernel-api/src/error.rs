//! Conversion from the C ABI `ayaka_status_t` into Rust errors.

use core::ffi::CStr;
use core::fmt;

use crate::ffi::{AyakaStatus, AyakaStatusCode};

/// A failed status returned by a native kernel entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError {
    pub code: AyakaStatusCode,
    pub message: String,
}

impl fmt::Display for KernelError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "kernel error {:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for KernelError {}

impl AyakaStatus {
    /// Convert into a `Result`, copying the message out of the C string.
    ///
    /// The native side only ever sets `message` to a pointer with static
    /// storage duration (string literals or `cudaGetErrorString`), so reading
    /// it here is sound even after the call returns.
    pub fn into_result(self) -> Result<(), KernelError> {
        if self.code == AyakaStatusCode::Ok {
            return Ok(());
        }
        let message = if self.message.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(self.message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(KernelError {
            code: self.code,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ok_status_into_result() {
        let status = AyakaStatus {
            code: AyakaStatusCode::Ok,
            message: core::ptr::null(),
        };
        assert!(status.into_result().is_ok());
    }

    #[test]
    fn test_error_status_carries_message() {
        let status = AyakaStatus {
            code: AyakaStatusCode::ShapeError,
            message: c"shape mismatch".as_ptr(),
        };
        let err = status.into_result().unwrap_err();
        assert_eq!(err.code, AyakaStatusCode::ShapeError);
        assert_eq!(err.message, "shape mismatch");
    }

    #[test]
    fn test_error_status_null_message() {
        let status = AyakaStatus {
            code: AyakaStatusCode::InternalError,
            message: core::ptr::null(),
        };
        let err = status.into_result().unwrap_err();
        assert_eq!(err.code, AyakaStatusCode::InternalError);
        assert!(err.message.is_empty());
    }
}
