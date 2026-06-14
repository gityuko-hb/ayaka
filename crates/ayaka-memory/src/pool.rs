use std::sync::Mutex;

use crate::{DeviceSpan, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PoolBlock {
    pub span: DeviceSpan,
    pub class_bytes: usize,
    pub requested_bytes: usize,
    offset: usize,
    class_index: usize,
}

pub struct SizeClassPool {
    base: DeviceSpan,
    classes: Vec<PoolClass>,
    tail: Mutex<usize>,
}

struct PoolClass {
    bytes: usize,
    free_offsets: Mutex<Vec<usize>>,
}

impl SizeClassPool {
    pub fn new(
        base: DeviceSpan,
        min_class: usize,
        max_class: usize,
    ) -> Result<Self> {
        if min_class == 0 || !min_class.is_power_of_two() {
            return Err(crate::error::invalid(format!(
                "min_class must be a non-zero power of two, got {min_class}"
            )));
        }
        if max_class < min_class || !max_class.is_power_of_two() {
            return Err(crate::error::invalid(format!(
                "max_class must be a power of two >= min_class, got {max_class}"
            )));
        }

        let mut classes = Vec::new();
        let mut bytes = min_class;
        loop {
            classes.push(PoolClass {
                bytes,
                free_offsets: Mutex::new(Vec::new()),
            });
            if bytes == max_class {
                break;
            }
            bytes = bytes
                .checked_mul(2)
                .ok_or_else(|| crate::error::overflow("pool class size overflow"))?;
        }

        Ok(Self {
            base,
            classes,
            tail: Mutex::new(0),
        })
    }

    pub fn alloc(
        &self,
        bytes: usize,
    ) -> Result<PoolBlock> {
        if bytes == 0 {
            return Err(crate::error::invalid(
                "pool allocation bytes must be non-zero",
            ));
        }
        let class_index = self.class_index(bytes)?;
        let class = &self.classes[class_index];

        if let Some(offset) = class
            .free_offsets
            .lock()
            .map_err(|_| crate::error::invalid("pool free list lock poisoned"))?
            .pop()
        {
            return Ok(PoolBlock {
                span: self.base.checked_subspan(offset, class.bytes)?,
                class_bytes: class.bytes,
                requested_bytes: bytes,
                offset,
                class_index,
            });
        }

        let mut tail = self
            .tail
            .lock()
            .map_err(|_| crate::error::invalid("pool tail lock poisoned"))?;
        let offset = align_up(*tail, class.bytes)?;
        let end = offset
            .checked_add(class.bytes)
            .ok_or_else(|| crate::error::overflow("pool allocation end offset overflow"))?;
        if end > self.base.len {
            return Err(crate::error::pool_exhausted(
                class.bytes,
                bytes,
                self.base.len.saturating_sub(offset),
            ));
        }
        *tail = end;
        Ok(PoolBlock {
            span: self.base.checked_subspan(offset, class.bytes)?,
            class_bytes: class.bytes,
            requested_bytes: bytes,
            offset,
            class_index,
        })
    }

    pub fn free(
        &self,
        block: PoolBlock,
    ) -> Result<()> {
        let class = self
            .classes
            .get(block.class_index)
            .ok_or_else(|| {
                crate::error::invalid(format!("pool class {} is invalid", block.class_index))
            })?;
        if class.bytes != block.class_bytes {
            return Err(crate::error::invalid(format!(
                "pool block class size {} does not match pool class {}",
                block.class_bytes, class.bytes
            )));
        }
        class
            .free_offsets
            .lock()
            .map_err(|_| crate::error::invalid("pool free list lock poisoned"))?
            .push(block.offset);
        Ok(())
    }

    fn class_index(
        &self,
        bytes: usize,
    ) -> Result<usize> {
        self.classes
            .iter()
            .position(|class| bytes <= class.bytes)
            .ok_or_else(|| {
                crate::error::invalid(format!(
                    "pool allocation request {bytes} exceeds max class {}",
                    self.classes.last().map(|c| c.bytes).unwrap_or(0)
                ))
            })
    }
}

fn align_up(
    value: usize,
    align: usize,
) -> Result<usize> {
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| crate::error::overflow("pool align_up overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_to_power_of_two_class() {
        let pool = SizeClassPool::new(DeviceSpan::new(0x1000, 4096), 256, 1024).unwrap();
        let block = pool.alloc(300).unwrap();
        assert_eq!(block.class_bytes, 512);
        assert_eq!(block.span.len, 512);
    }

    #[test]
    fn freed_block_is_reused() {
        let pool = SizeClassPool::new(DeviceSpan::new(0x1000, 4096), 256, 1024).unwrap();
        let block = pool.alloc(200).unwrap();
        let ptr = block.span.ptr;
        pool.free(block).unwrap();
        assert_eq!(pool.alloc(200).unwrap().span.ptr, ptr);
    }

    #[test]
    fn exhaustion_is_typed_error() {
        use ayaka_error::ErrorKind;
        let pool = SizeClassPool::new(DeviceSpan::new(0x1000, 512), 256, 512).unwrap();
        let _a = pool.alloc(400).unwrap();
        let err = pool.alloc(400).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.message().contains("pool exhausted"));
    }
}
