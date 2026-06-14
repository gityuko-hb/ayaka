use crate::{DeviceSpan, Result};

pub struct DeviceArena {
    base: DeviceSpan,
    cursor: usize,
}

impl DeviceArena {
    pub fn new(base: DeviceSpan) -> Self {
        Self { base, cursor: 0 }
    }

    pub fn base(&self) -> DeviceSpan {
        self.base
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn remaining(&self) -> usize {
        self.base.len.saturating_sub(self.cursor)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn alloc(
        &mut self,
        bytes: usize,
        align: usize,
    ) -> Result<DeviceSpan> {
        if bytes == 0 {
            return Ok(DeviceSpan::new(self.base.ptr + self.cursor as u64, 0));
        }
        let offset = align_up(self.cursor, align)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| crate::error::overflow("arena allocation end offset overflow"))?;
        if end > self.base.len {
            return Err(crate::error::arena_exhausted(
                bytes,
                self.base.len.saturating_sub(offset),
            ));
        }
        let span = self.base.checked_subspan(offset, bytes)?;
        self.cursor = end;
        Ok(span)
    }
}

fn align_up(
    value: usize,
    align: usize,
) -> Result<usize> {
    if align == 0 || !align.is_power_of_two() {
        return Err(crate::error::invalid(format!(
            "arena alignment must be a non-zero power of two, got {align}"
        )));
    }
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| crate::error::overflow("arena align_up overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_alloc_aligns_offsets() {
        let mut arena = DeviceArena::new(DeviceSpan::new(0x1000, 1024));
        let a = arena.alloc(1, 256).unwrap();
        let b = arena.alloc(1, 256).unwrap();
        assert_eq!(a.ptr, 0x1000);
        assert_eq!(b.ptr, 0x1100);
    }

    #[test]
    fn reset_reuses_space() {
        let mut arena = DeviceArena::new(DeviceSpan::new(0x1000, 64));
        let first = arena.alloc(32, 16).unwrap();
        arena.reset();
        let second = arena.alloc(32, 16).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn exhaustion_is_typed_error() {
        use ayaka_error::ErrorKind;
        let mut arena = DeviceArena::new(DeviceSpan::new(0x1000, 64));
        let err = arena.alloc(65, 1).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.message().contains("arena exhausted"));
    }
}
