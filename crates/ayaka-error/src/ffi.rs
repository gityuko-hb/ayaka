//! C ABI status struct and conversions.

use core::ffi::c_char;

use crate::{AyakaError, ErrorKind, Result, StatusCode};

/// C ABI boundary status.
///
/// Mirrors `ayaka_status_t` in `native/include/ayaka/status.h`. The `message`
/// pointer must always point to a static string literal, a `cudaGetErrorString`
/// result, or be null. The receiver must not free it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AyakaStatus {
    pub code: i32,
    pub message: *const c_char,
}

impl AyakaStatus {
    /// Returns a successful status with no message.
    pub const fn ok() -> Self {
        Self {
            code: StatusCode::Ok.as_i32(),
            message: core::ptr::null(),
        }
    }

    /// Returns an error status from a static C string.
    ///
    /// # Safety
    ///
    /// `message` must be a valid pointer to a null-terminated static C string
    /// or null. The pointer must remain valid for the lifetime of the returned
    /// status.
    pub const unsafe fn from_static(code: StatusCode, message: *const c_char) -> Self {
        Self {
            code: code.as_i32(),
            message,
        }
    }

    /// Converts a Rust `Result<()>` into an `AyakaStatus`.
    ///
    /// On success the message is null. On error the message points to a static
    /// description generated from the error; the native side must not free it.
    pub fn from_result(result: &Result<()>) -> Self {
        match result {
            Ok(()) => Self::ok(),
            Err(err) => Self {
                code: err.status_code().as_i32(),
                message: static_message(err).as_ptr() as *const c_char,
            },
        }
    }

    /// Converts this status into a Rust `Result<()>`.
    ///
    /// # Safety
    ///
    /// `self.message` must be a pointer to a static null-terminated C string or
    /// null. The resulting `AyakaError` copies the message bytes, so the
    /// pointer does not need to outlive the call.
    pub unsafe fn into_result(self) -> Result<()> {
        let code = StatusCode::from_i32(self.code);
        if code == StatusCode::Ok {
            return Ok(());
        }
        let message = if self.message.is_null() {
            String::new()
        } else {
            unsafe {
                core::ffi::CStr::from_ptr(self.message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(AyakaError::new(code.into_kind(), message))
    }
}

fn static_message(err: &AyakaError) -> &'static [u8] {
    // AyakaStatus only crosses the FFI boundary with a static placeholder
    // because the Rust error message is heap-allocated. Callers that need the
    // full message should retrieve it through an out-parameter or a dedicated
    // diagnostic API. For now we return a static tag per kind.
    match err.kind() {
        ErrorKind::OutOfMemory | ErrorKind::Memory => b"out of memory\0",
        ErrorKind::InvalidArgument => b"invalid argument\0",
        ErrorKind::Shape => b"shape error\0",
        ErrorKind::DType => b"dtype error\0",
        ErrorKind::Device | ErrorKind::Cuda => b"device error\0",
        ErrorKind::KernelLaunch => b"kernel launch error\0",
        ErrorKind::ContextLengthExceeded => b"context length exceeded\0",
        ErrorKind::NotFound => b"not found\0",
        ErrorKind::AbiMismatch => b"abi mismatch\0",
        ErrorKind::Cancelled => b"cancelled\0",
        _ => b"internal error\0",
    }
}

impl From<StatusCode> for ErrorKind {
    fn from(code: StatusCode) -> Self {
        code.into_kind()
    }
}

impl StatusCode {
    /// Maps a status code back to the closest `ErrorKind`.
    pub const fn into_kind(self) -> ErrorKind {
        match self {
            StatusCode::Ok => ErrorKind::Internal,
            StatusCode::InvalidArgument => ErrorKind::InvalidArgument,
            StatusCode::DTypeError => ErrorKind::DType,
            StatusCode::ShapeError => ErrorKind::Shape,
            StatusCode::DeviceError => ErrorKind::Device,
            StatusCode::Unsupported => ErrorKind::Unsupported,
            StatusCode::OutOfMemory => ErrorKind::OutOfMemory,
            StatusCode::ContextLengthExceeded => ErrorKind::ContextLengthExceeded,
            StatusCode::NotFound => ErrorKind::NotFound,
            StatusCode::AbiMismatch => ErrorKind::AbiMismatch,
            StatusCode::KernelLaunchError => ErrorKind::KernelLaunch,
            StatusCode::StreamError => ErrorKind::Cuda,
            StatusCode::InternalError => ErrorKind::Internal,
            StatusCode::Cancelled => ErrorKind::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext::{OptionExt, ResultExt};
    use crate::ErrorKind;

    fn maybe_ok(ok: bool) -> core::result::Result<i32, &'static str> {
        if ok {
            Ok(42)
        } else {
            Err("boom")
        }
    }

    #[test]
    fn result_ext_context() {
        let r: Result<i32> = maybe_ok(false).ayaka_context(ErrorKind::Shape, "parse");
        let err = r.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Shape);
        assert!(err.message().contains("parse"));
        assert!(err.message().contains("boom"));
    }

    #[test]
    fn result_ext_with_context() {
        let r: Result<i32> = maybe_ok(false)
            .ayaka_with_context(ErrorKind::Memory, |e| alloc::format!("alloc: {e}"));
        let err = r.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Memory);
        assert!(err.message().contains("alloc: boom"));
    }

    #[test]
    fn option_ext_ok_or_ayaka() {
        let opt: Option<i32> = None;
        let r = opt.ok_or_ayaka(ErrorKind::NotFound, "missing tensor");
        let err = r.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn ayaka_status_ok() {
        let status = AyakaStatus::from_result(&Ok(()));
        assert_eq!(status.code, StatusCode::Ok.as_i32());
        assert!(status.message.is_null());
    }

    #[test]
    fn ayaka_status_error() {
        let err: Result<()> = Err(AyakaError::out_of_memory("no device memory"));
        let status = AyakaStatus::from_result(&err);
        assert_eq!(status.code, StatusCode::OutOfMemory.as_i32());
        assert!(!status.message.is_null());
        unsafe {
            let msg = core::ffi::CStr::from_ptr(status.message)
                .to_string_lossy()
                .into_owned();
            assert!(msg.contains("out of memory"));
        }
    }

    #[test]
    fn ayaka_status_into_result_ok() {
        let status = AyakaStatus::ok();
        unsafe {
            assert!(status.into_result().is_ok());
        }
    }

    #[test]
    fn ayaka_status_into_result_error() {
        let msg = c"shape mismatch".as_ptr();
        let status = unsafe { AyakaStatus::from_static(StatusCode::ShapeError, msg) };
        unsafe {
            let err = status.into_result().unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Shape);
            assert!(err.message().contains("shape mismatch"));
        }
    }
}
