use crate::{MemoryError, Result};

/// Non-owning byte span in device memory.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DeviceSpan {
    pub ptr: u64,
    pub len: usize,
}

impl DeviceSpan {
    pub const fn new(ptr: u64, len: usize) -> Self {
        Self { ptr, len }
    }

    pub const fn empty() -> Self {
        Self { ptr: 0, len: 0 }
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn end_ptr(self) -> Result<u64> {
        self.ptr
            .checked_add(self.len as u64)
            .ok_or_else(|| MemoryError::overflow("device span end pointer overflow"))
    }

    pub fn checked_subspan(self, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| MemoryError::overflow("device subspan offset overflow"))?;
        if end > self.len {
            return Err(MemoryError::invalid(format!(
                "subspan [{offset}, {end}) exceeds span length {}",
                self.len
            )));
        }
        let ptr = self
            .ptr
            .checked_add(offset as u64)
            .ok_or_else(|| MemoryError::overflow("device subspan pointer overflow"))?;
        Ok(Self { ptr, len })
    }

    pub fn is_aligned_to(self, align: usize) -> bool {
        align != 0 && align.is_power_of_two() && self.ptr % align as u64 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subspan_uses_byte_offsets() {
        let span = DeviceSpan::new(0x1000, 1024);
        assert_eq!(span.end_ptr().unwrap(), 0x1400);
        assert_eq!(span.checked_subspan(256, 128).unwrap(), DeviceSpan::new(0x1100, 128));
    }

    #[test]
    fn rejects_out_of_range_subspan() {
        let span = DeviceSpan::new(0x1000, 32);
        assert!(span.checked_subspan(16, 32).is_err());
    }

    #[test]
    fn alignment_is_checked_in_bytes() {
        assert!(DeviceSpan::new(0x2000, 64).is_aligned_to(256));
        assert!(!DeviceSpan::new(0x2080, 64).is_aligned_to(256));
        assert!(!DeviceSpan::new(0x2000, 64).is_aligned_to(0));
    }
}
