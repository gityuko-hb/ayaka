use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{DeviceSpan, MemoryError, Result};

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SlabSlot(u32);

impl SlabSlot {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

pub struct SlabAllocator {
    base: DeviceSpan,
    slot_bytes: usize,
    num_slots: usize,
    words: Box<[AtomicU64]>,
    cursor_word: AtomicUsize,
    free_count: AtomicUsize,
}

impl SlabAllocator {
    pub fn new(
        base: DeviceSpan,
        slot_bytes: usize,
        num_slots: usize,
    ) -> Result<Self> {
        if slot_bytes == 0 {
            return Err(MemoryError::invalid("slab slot_bytes must be non-zero"));
        }
        if num_slots == 0 {
            return Err(MemoryError::invalid("slab num_slots must be non-zero"));
        }
        let required = slot_bytes
            .checked_mul(num_slots)
            .ok_or_else(|| MemoryError::overflow("slab byte size overflow"))?;
        if required > base.len {
            return Err(MemoryError::invalid(format!(
                "slab requires {required} bytes but base span has {} bytes",
                base.len
            )));
        }

        let word_count = num_slots.div_ceil(64);
        let mut words = Vec::with_capacity(word_count);
        for word_index in 0..word_count {
            let first_slot = word_index * 64;
            let remaining = num_slots.saturating_sub(first_slot);
            let valid_bits = remaining.min(64);
            let valid_mask = if valid_bits == 64 {
                u64::MAX
            } else {
                (1u64 << valid_bits) - 1
            };
            words.push(AtomicU64::new(!valid_mask));
        }

        Ok(Self {
            base,
            slot_bytes,
            num_slots,
            words: words.into_boxed_slice(),
            cursor_word: AtomicUsize::new(0),
            free_count: AtomicUsize::new(num_slots),
        })
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub fn free_count(&self) -> usize {
        self.free_count.load(Ordering::Acquire)
    }

    pub fn alloc(&self) -> Result<SlabSlot> {
        let word_count = self.words.len();
        let start = self.cursor_word.load(Ordering::Relaxed) % word_count;
        for probe in 0..word_count {
            let word_index = (start + probe) % word_count;
            let word = &self.words[word_index];
            loop {
                let current = word.load(Ordering::Acquire);
                let free_bits = !current;
                if free_bits == 0 {
                    break;
                }
                let bit = free_bits.trailing_zeros();
                let mask = 1u64 << bit;
                let next = current | mask;
                match word.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => {
                        let slot = word_index * 64 + bit as usize;
                        self.cursor_word
                            .store(word_index, Ordering::Relaxed);
                        self.free_count.fetch_sub(1, Ordering::AcqRel);
                        return Ok(SlabSlot(slot as u32));
                    },
                    Err(_) => continue,
                }
            }
        }
        Err(MemoryError::SlabExhausted {
            num_slots: self.num_slots,
        })
    }

    pub fn free(
        &self,
        slot: SlabSlot,
    ) -> Result<()> {
        let raw = slot.raw() as usize;
        if raw >= self.num_slots {
            return Err(MemoryError::InvalidSlot {
                slot: slot.raw(),
                num_slots: self.num_slots,
            });
        }
        let word_index = raw / 64;
        let bit = raw % 64;
        let mask = 1u64 << bit;
        let previous = self.words[word_index].fetch_and(!mask, Ordering::AcqRel);
        if previous & mask == 0 {
            return Err(MemoryError::SlotAlreadyFree { slot: slot.raw() });
        }
        self.cursor_word
            .store(word_index, Ordering::Release);
        self.free_count.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    pub fn slot_span(
        &self,
        slot: SlabSlot,
    ) -> Result<DeviceSpan> {
        let raw = slot.raw() as usize;
        if raw >= self.num_slots {
            return Err(MemoryError::InvalidSlot {
                slot: slot.raw(),
                num_slots: self.num_slots,
            });
        }
        let offset = raw
            .checked_mul(self.slot_bytes)
            .ok_or_else(|| MemoryError::overflow("slab slot offset overflow"))?;
        self.base.checked_subspan(offset, self.slot_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn allocates_fixed_slots_and_reuses_freed_slot() {
        let slab = SlabAllocator::new(DeviceSpan::new(0x1000, 4096), 1024, 4).unwrap();
        let a = slab.alloc().unwrap();
        assert_eq!(slab.slot_span(a).unwrap(), DeviceSpan::new(0x1000, 1024));
        slab.free(a).unwrap();
        let b = slab.alloc().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn double_free_is_rejected() {
        let slab = SlabAllocator::new(DeviceSpan::new(0x1000, 2048), 1024, 2).unwrap();
        let slot = slab.alloc().unwrap();
        slab.free(slot).unwrap();
        assert!(slab.free(slot).is_err());
    }

    #[test]
    fn full_slab_returns_typed_error() {
        let slab = SlabAllocator::new(DeviceSpan::new(0x1000, 2048), 1024, 2).unwrap();
        let _a = slab.alloc().unwrap();
        let _b = slab.alloc().unwrap();
        assert!(matches!(
            slab.alloc().unwrap_err(),
            MemoryError::SlabExhausted { num_slots: 2 }
        ));
    }

    #[test]
    fn multithread_alloc_never_hands_out_same_slot_twice() {
        let slab =
            Arc::new(SlabAllocator::new(DeviceSpan::new(0x1000, 64 * 128), 64, 128).unwrap());
        let seen = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let slab = Arc::clone(&slab);
            let seen = Arc::clone(&seen);
            handles.push(thread::spawn(move || {
                while let Ok(slot) = slab.alloc() {
                    let inserted = seen.lock().unwrap().insert(slot.raw());
                    assert!(inserted, "slot {} allocated twice", slot.raw());
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(seen.lock().unwrap().len(), 128);
        assert_eq!(slab.free_count(), 0);
    }
}
