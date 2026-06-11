//! Thin wrappers over native CUDA memory C ABI entry points.

#[cfg(feature = "cuda")]
use core::ffi::c_void;

#[cfg(feature = "cuda")]
use crate::error::KernelError;
#[cfg(feature = "cuda")]
use crate::ffi::{self, AyakaMemInfo, AyakaMemcpyKind, AyakaStream};

#[cfg(feature = "cuda")]
pub unsafe fn device_alloc(bytes: usize, device_ordinal: i32) -> Result<*mut c_void, KernelError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_device_alloc(&mut out, bytes, device_ordinal) }.into_result()?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub unsafe fn device_free(ptr: *mut c_void, device_ordinal: i32) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_device_free(ptr, device_ordinal) }.into_result()
}

#[cfg(feature = "cuda")]
pub unsafe fn device_alloc_async(
    bytes: usize,
    device_ordinal: i32,
    stream: AyakaStream,
) -> Result<*mut c_void, KernelError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_device_alloc_async(&mut out, bytes, device_ordinal, stream) }
        .into_result()?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub unsafe fn device_free_async(
    ptr: *mut c_void,
    device_ordinal: i32,
    stream: AyakaStream,
) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_device_free_async(ptr, device_ordinal, stream) }.into_result()
}

#[cfg(feature = "cuda")]
pub unsafe fn host_pinned_alloc(bytes: usize) -> Result<*mut c_void, KernelError> {
    let mut out = core::ptr::null_mut();
    unsafe { ffi::ayaka_mem_host_pinned_alloc(&mut out, bytes) }.into_result()?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub unsafe fn host_pinned_free(ptr: *mut c_void) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_host_pinned_free(ptr) }.into_result()
}

#[cfg(feature = "cuda")]
pub unsafe fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: AyakaMemcpyKind,
    stream: AyakaStream,
) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_memcpy_async(dst, src, bytes, kind, stream) }.into_result()
}

#[cfg(feature = "cuda")]
pub unsafe fn memset_async(
    dst: *mut c_void,
    value: i32,
    bytes: usize,
    stream: AyakaStream,
) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_memset_async(dst, value, bytes, stream) }.into_result()
}

#[cfg(feature = "cuda")]
pub fn get_info(device_ordinal: i32) -> Result<AyakaMemInfo, KernelError> {
    let mut info = AyakaMemInfo {
        free_bytes: 0,
        total_bytes: 0,
    };
    unsafe { ffi::ayaka_mem_get_info(&mut info, device_ordinal) }.into_result()?;
    Ok(info)
}

#[cfg(feature = "cuda")]
pub fn stream_synchronize(stream: AyakaStream) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_stream_synchronize(stream) }.into_result()
}

#[cfg(feature = "cuda")]
pub fn pool_supported(device_ordinal: i32) -> Result<bool, KernelError> {
    let mut supported = 0;
    unsafe { ffi::ayaka_mem_pool_supported(&mut supported, device_ordinal) }.into_result()?;
    Ok(supported != 0)
}

#[cfg(feature = "cuda")]
pub fn pool_trim(min_bytes_to_keep: usize, device_ordinal: i32) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_mem_pool_trim(min_bytes_to_keep, device_ordinal) }.into_result()
}
