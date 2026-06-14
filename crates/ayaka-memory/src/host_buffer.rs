use core::ptr::NonNull;
use std::alloc::{Layout, alloc_zeroed, dealloc};

use crate::Result;

const HOST_ALIGNMENT: usize = 64;

/// Cache-line-aligned host byte buffer with stable address and capacity.
pub struct HostBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Option<Layout>,
}

impl HostBuffer {
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                layout: None,
            });
        }

        let layout = Layout::from_size_align(len, HOST_ALIGNMENT)
            .map_err(|_| crate::error::invalid(format!("invalid host layout for {len} bytes")))?;
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or(crate::error::host_alloc(len))?;
        Ok(Self {
            ptr,
            len,
            layout: Some(layout),
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_64_byte_aligned_and_zeroed() {
        let buffer = HostBuffer::new(129).unwrap();
        assert_eq!(buffer.len(), 129);
        assert_eq!(buffer.as_ptr() as usize % 64, 0);
        assert!(buffer.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn mutable_slice_writes_through() {
        let mut buffer = HostBuffer::new(4).unwrap();
        buffer
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn zero_length_buffer_is_allowed() {
        let buffer = HostBuffer::new(0).unwrap();
        assert!(buffer.is_empty());
        assert!(buffer.as_slice().is_empty());
    }
}
