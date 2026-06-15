//! Per-sequence page table and slot-mapping construction.
//!
//! Maps each sequence to its ordered list of physical pages (logical block
//! index → [`PageId`]) and turns token positions into the flat `slot_mapping`
//! the `append_kv` kernel consumes.

use std::collections::HashMap;

use ayaka_core::id::{PageId, SequenceId};

/// Tokens per page; the FlashInfer/vLLM default. Configurable per table.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Ordered physical pages per sequence.
#[derive(Debug)]
pub struct PageTable {
    block_size: usize,
    pages: HashMap<SequenceId, Vec<PageId>>,
}

impl PageTable {
    /// Empty table with the given page size in tokens.
    ///
    /// # Panics
    /// If `block_size == 0`.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "PageTable: block_size must be > 0");
        Self {
            block_size,
            pages: HashMap::new(),
        }
    }

    /// Empty table using [`DEFAULT_BLOCK_SIZE`].
    pub fn with_default_block_size() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }

    /// Tokens per page.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Append a physical page to `seq`'s page list (its next logical block).
    pub fn append_page(
        &mut self,
        seq: SequenceId,
        page: PageId,
    ) {
        self.pages.entry(seq).or_default().push(page);
    }

    /// The pages backing `seq` in logical-block order (empty if unknown).
    pub fn pages(
        &self,
        seq: SequenceId,
    ) -> &[PageId] {
        self.pages.get(&seq).map_or(&[], Vec::as_slice)
    }

    /// Flat slot index for each token `position` of `seq`, suitable as the
    /// kernel's `slot_mapping`:
    /// `slot = page[pos / block_size] * block_size + pos % block_size`.
    ///
    /// # Panics
    /// If a position falls in a block `seq` has no page for (i.e. the caller
    /// has not allocated enough pages).
    pub fn build_slot_mapping(
        &self,
        seq: SequenceId,
        positions: &[usize],
    ) -> Vec<i64> {
        let pages = self.pages(seq);
        positions
            .iter()
            .map(|&pos| {
                let block = pos / self.block_size;
                let offset = pos % self.block_size;
                let page = pages.get(block).unwrap_or_else(|| {
                    panic!(
                        "PageTable: position {pos} needs block {block} but {seq} has {} page(s)",
                        pages.len()
                    )
                });
                page.raw() as i64 * self.block_size as i64 + offset as i64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_block_size_is_sixteen() {
        let t = PageTable::with_default_block_size();
        assert_eq!(t.block_size(), DEFAULT_BLOCK_SIZE);
        assert_eq!(DEFAULT_BLOCK_SIZE, 16);
    }

    #[test]
    fn append_and_read_pages() {
        let mut t = PageTable::new(4);
        let seq = SequenceId::new(1);
        assert!(t.pages(seq).is_empty());
        t.append_page(seq, PageId::new(10));
        t.append_page(seq, PageId::new(20));
        assert_eq!(t.pages(seq), &[PageId::new(10), PageId::new(20)]);
    }

    #[test]
    fn slot_mapping_within_one_block() {
        let mut t = PageTable::new(8);
        let seq = SequenceId::new(7);
        t.append_page(seq, PageId::new(3));
        // page 3, block size 8 → slots 24..
        assert_eq!(t.build_slot_mapping(seq, &[0, 1, 7]), vec![24, 25, 31]);
    }

    #[test]
    fn slot_mapping_crosses_block_boundaries() {
        let mut t = PageTable::new(4);
        let seq = SequenceId::new(2);
        for p in [10u32, 20, 30] {
            t.append_page(seq, PageId::new(p));
        }
        // block 0 → page 10, block 1 → page 20, block 2 → page 30.
        let slots = t.build_slot_mapping(seq, &[0, 3, 4, 7, 8, 11]);
        assert_eq!(slots, vec![40, 43, 80, 83, 120, 123]);
    }

    #[test]
    #[should_panic(expected = "needs block")]
    fn slot_mapping_panics_on_unallocated_block() {
        let mut t = PageTable::new(4);
        let seq = SequenceId::new(5);
        t.append_page(seq, PageId::new(0));
        let _ = t.build_slot_mapping(seq, &[4]); // block 1 not allocated
    }
}
