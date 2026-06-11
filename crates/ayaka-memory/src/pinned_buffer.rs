#![cfg(feature = "cuda")]

use core::ffi::c_void;
use core::ptr::NonNull;

use ayaka_kernel_api::mem;

use crate::{MemoryError, Result};

pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

impl PinnedBuffer {
    pub fn new(len: usize) -> Result<Self> {
        let ptr = if len == 0 {
            NonNull::dangling()
        } else {
            let raw = unsafe { mem::host_pinned_alloc(len)? };
            NonNull::new(raw.cast::<u8>())
                .ok_or_else(|| MemoryError::invalid("pinned allocation returned null"))?
        };
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
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

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        if let Err(err) = unsafe { mem::host_pinned_free(self.ptr.as_ptr() as *mut c_void) } {
            tracing::error!(error = %err, "leaking pinned host buffer after failed free");
        }
    }
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

pub struct PinnedRing {
    buffer: PinnedBuffer,
    slot_count: usize,
    slot_bytes: usize,
}

impl PinnedRing {
    pub fn new(
        slot_count: usize,
        slot_bytes: usize,
    ) -> Result<Self> {
        if slot_count == 0 {
            return Err(MemoryError::invalid(
                "pinned ring slot_count must be non-zero",
            ));
        }
        if slot_bytes == 0 {
            return Err(MemoryError::invalid(
                "pinned ring slot_bytes must be non-zero",
            ));
        }
        let total = slot_count
            .checked_mul(slot_bytes)
            .ok_or_else(|| MemoryError::overflow("pinned ring total byte count overflow"))?;
        Ok(Self {
            buffer: PinnedBuffer::new(total)?,
            slot_count,
            slot_bytes,
        })
    }

    pub fn acquire(
        &mut self,
        step: usize,
    ) -> PinnedSlot<'_> {
        let index = step % self.slot_count;
        let start = index * self.slot_bytes;
        let end = start + self.slot_bytes;
        PinnedSlot {
            index,
            bytes: &mut self.buffer.as_mut_slice()[start..end],
            cursor: 0,
        }
    }
}

pub struct PinnedSlot<'a> {
    index: usize,
    bytes: &'a mut [u8],
    cursor: usize,
}

impl<'a> PinnedSlot<'a> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }

    pub fn bump_write(
        &mut self,
        data: &[u8],
        align: usize,
    ) -> Result<&mut [u8]> {
        let start = align_up(self.cursor, align)?;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| MemoryError::overflow("pinned slot write offset overflow"))?;
        if end > self.bytes.len() {
            return Err(MemoryError::invalid(format!(
                "pinned slot write of {} bytes exceeds remaining {} bytes",
                data.len(),
                self.bytes.len().saturating_sub(start)
            )));
        }
        self.bytes[start..end].copy_from_slice(data);
        self.cursor = end;
        Ok(&mut self.bytes[start..end])
    }
}

fn align_up(
    value: usize,
    align: usize,
) -> Result<usize> {
    if align == 0 || !align.is_power_of_two() {
        return Err(MemoryError::invalid(format!(
            "alignment must be a non-zero power of two, got {align}"
        )));
    }
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| MemoryError::overflow("align_up overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_buffer_is_host_accessible() {
        let mut buffer = PinnedBuffer::new(4).unwrap();
        buffer
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn ring_reuses_step_mod_slot_count() {
        let mut ring = PinnedRing::new(2, 16).unwrap();
        assert_eq!(ring.acquire(0).index(), 0);
        assert_eq!(ring.acquire(1).index(), 1);
        assert_eq!(ring.acquire(2).index(), 0);
    }

    #[test]
    fn slot_bump_write_aligns_and_copies() {
        let mut ring = PinnedRing::new(2, 32).unwrap();
        let mut slot = ring.acquire(0);
        assert_eq!(slot.bump_write(&[1, 2, 3], 1).unwrap(), &[1, 2, 3]);
        assert_eq!(slot.bump_write(&[4], 8).unwrap(), &[4]);
    }
}
