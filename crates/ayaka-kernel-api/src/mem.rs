//! Thin wrappers over native CUDA memory C ABI entry points.

#[cfg(feature = "cuda")]
use core::ffi::c_void;

#[cfg(feature = "cuda")]
use crate::ffi::{self, AyakaMemInfo, AyakaMemcpyKind, AyakaStream};
#[cfg(feature = "cuda")]
use ayaka_error::AyakaError;

/// # Safety
///
/// `device_ordinal` must name an existing CUDA device. The returned pointer
/// must be released with [`device_free`] on the same device.
#[cfg(feature = "cuda")]
pub unsafe fn device_alloc(
    bytes: usize,
    device_ordinal: i32,
) -> Result<*mut c_void, AyakaError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_device_alloc(&mut out, bytes, device_ordinal) }.into_result()?;
    Ok(out)
}

/// # Safety
///
/// `ptr` must come from [`device_alloc`] on `device_ordinal`, must not be
/// freed twice, and no kernel or copy may still use it.
#[cfg(feature = "cuda")]
pub unsafe fn device_free(
    ptr: *mut c_void,
    device_ordinal: i32,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_device_free(ptr, device_ordinal) }
        .into_result()
        .map_err(AyakaError::from)
}

/// # Safety
///
/// `device_ordinal` must name an existing CUDA device and `stream` must be a
/// valid CUDA stream on it. The returned pointer is usable only as ordered by
/// `stream` and must be released with [`device_free_async`].
#[cfg(feature = "cuda")]
pub unsafe fn device_alloc_async(
    bytes: usize,
    device_ordinal: i32,
    stream: AyakaStream,
) -> Result<*mut c_void, AyakaError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_device_alloc_async(&mut out, bytes, device_ordinal, stream) }
        .into_result()?;
    Ok(out)
}

/// # Safety
///
/// `ptr` must come from [`device_alloc_async`] on `device_ordinal`, must not
/// be freed twice, and all work using it must be ordered before this free on
/// `stream`, which must be a valid CUDA stream on the device.
#[cfg(feature = "cuda")]
pub unsafe fn device_free_async(
    ptr: *mut c_void,
    device_ordinal: i32,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_device_free_async(ptr, device_ordinal, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// # Safety
///
/// The returned pointer must be released with [`host_pinned_free`].
#[cfg(feature = "cuda")]
pub unsafe fn host_pinned_alloc(bytes: usize) -> Result<*mut c_void, AyakaError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_host_pinned_alloc(&mut out, bytes) }.into_result()?;
    Ok(out)
}

/// # Safety
///
/// `ptr` must come from [`host_pinned_alloc`], must not be freed twice, and
/// no asynchronous copy may still use it.
#[cfg(feature = "cuda")]
pub unsafe fn host_pinned_free(ptr: *mut c_void) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_host_pinned_free(ptr) }
        .into_result()
        .map_err(AyakaError::from)
}

/// # Safety
///
/// `dst` and `src` must be live allocations of at least `bytes` whose sides
/// match `kind` (host or device), staying valid until the copy completes on
/// `stream`, which must be a valid CUDA stream.
#[cfg(feature = "cuda")]
pub unsafe fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: AyakaMemcpyKind,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_memcpy_async(dst, src, bytes, kind, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// # Safety
///
/// `dst` must be a live device allocation of at least `bytes`, staying valid
/// until the memset completes on `stream`, which must be a valid CUDA stream.
#[cfg(feature = "cuda")]
pub unsafe fn memset_async(
    dst: *mut c_void,
    value: i32,
    bytes: usize,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_memset_async(dst, value, bytes, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

#[cfg(feature = "cuda")]
pub fn get_info(device_ordinal: i32) -> Result<AyakaMemInfo, AyakaError> {
    let mut info = AyakaMemInfo {
        free_bytes: 0,
        total_bytes: 0,
    };
    unsafe { ffi::ayaka_mem_get_info(&mut info, device_ordinal) }.into_result()?;
    Ok(info)
}

// `AyakaStream` is an opaque handle the native side hands to the CUDA
// runtime; Rust never dereferences it. An invalid handle surfaces as a
// status error, not UB on this side.
#[cfg(feature = "cuda")]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn stream_synchronize(stream: AyakaStream) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_stream_synchronize(stream) }
        .into_result()
        .map_err(AyakaError::from)
}

#[cfg(feature = "cuda")]
pub fn pool_supported(device_ordinal: i32) -> Result<bool, AyakaError> {
    let mut supported = 0;
    unsafe { ffi::ayaka_mem_pool_supported(&mut supported, device_ordinal) }.into_result()?;
    Ok(supported != 0)
}

#[cfg(feature = "cuda")]
pub fn pool_trim(
    min_bytes_to_keep: usize,
    device_ordinal: i32,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_mem_pool_trim(min_bytes_to_keep, device_ordinal) }
        .into_result()
        .map_err(AyakaError::from)
}
