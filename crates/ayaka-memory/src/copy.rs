#![cfg(feature = "cuda")]

use core::ffi::c_void;

use ayaka_kernel_api::ffi::AyakaMemcpyKind;
use ayaka_kernel_api::mem;

use crate::{DeviceSpan, MemoryError, Result, StreamHandle};

pub fn h2d_async(
    dst: DeviceSpan,
    src: &[u8],
    stream: StreamHandle,
) -> Result<()> {
    if src.len() > dst.len {
        return Err(MemoryError::invalid(format!(
            "h2d copy source length {} exceeds destination length {}",
            src.len(),
            dst.len
        )));
    }
    unsafe {
        mem::memcpy_async(
            dst.ptr as usize as *mut c_void,
            src.as_ptr() as *const c_void,
            src.len(),
            AyakaMemcpyKind::HostToDevice,
            stream.as_ffi(),
        )?;
    }
    Ok(())
}

pub fn d2h_async(
    dst: &mut [u8],
    src: DeviceSpan,
    stream: StreamHandle,
) -> Result<()> {
    if dst.len() > src.len {
        return Err(MemoryError::invalid(format!(
            "d2h copy destination length {} exceeds source length {}",
            dst.len(),
            src.len
        )));
    }
    unsafe {
        mem::memcpy_async(
            dst.as_mut_ptr() as *mut c_void,
            src.ptr as usize as *const c_void,
            dst.len(),
            AyakaMemcpyKind::DeviceToHost,
            stream.as_ffi(),
        )?;
    }
    Ok(())
}

pub fn d2d_async(
    dst: DeviceSpan,
    src: DeviceSpan,
    bytes: usize,
    stream: StreamHandle,
) -> Result<()> {
    if bytes > dst.len || bytes > src.len {
        return Err(MemoryError::invalid(format!(
            "d2d copy length {bytes} exceeds dst {} or src {}",
            dst.len, src.len
        )));
    }
    unsafe {
        mem::memcpy_async(
            dst.ptr as usize as *mut c_void,
            src.ptr as usize as *const c_void,
            bytes,
            AyakaMemcpyKind::DeviceToDevice,
            stream.as_ffi(),
        )?;
    }
    Ok(())
}

pub fn memset_async(
    dst: DeviceSpan,
    value: u8,
    stream: StreamHandle,
) -> Result<()> {
    unsafe {
        mem::memset_async(
            dst.ptr as usize as *mut c_void,
            i32::from(value),
            dst.len,
            stream.as_ffi(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2d_rejects_oversized_source() {
        let dst = DeviceSpan::new(0x1000, 4);
        let src = [0u8; 8];
        assert!(h2d_async(dst, &src, StreamHandle::null()).is_err());
    }
}
