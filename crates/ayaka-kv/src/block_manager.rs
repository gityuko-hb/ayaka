//! Block manager: ties the physical [`PageAllocator`] to the per-sequence
//! [`PageTable`] and adds the sharing semantics the paged KV cache needs —
//! `fork` (share a parent's pages with a child via reference counts) and
//! copy-on-write (give a shared page back to a single owner before it is
//! mutated). Pure Rust; owns no device memory.

use ayaka_core::id::{PageId, SequenceId};

use crate::allocator::PageAllocator;
use crate::page_table::PageTable;

/// Outcome of [`BlockManager::ensure_writable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CowOutcome {
    /// The block was already uniquely owned; no copy needed.
    AlreadyOwned(PageId),
    /// The block was shared; a fresh page was allocated. The caller must copy
    /// the contents of `from` into `to` before writing.
    Copied { from: PageId, to: PageId },
}

/// Combines page allocation, per-sequence page tables, and ref-counted
/// sharing.
#[derive(Debug)]
pub struct BlockManager {
    allocator: PageAllocator,
    table: PageTable,
}

impl BlockManager {
    /// New manager over `num_blocks` physical pages with the given page size.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
    ) -> Self {
        Self {
            allocator: PageAllocator::new(num_blocks),
            table: PageTable::new(block_size),
        }
    }

    /// Tokens per page.
    pub fn block_size(&self) -> usize {
        self.table.block_size()
    }

    /// Free physical pages remaining.
    pub fn free_pages(&self) -> usize {
        self.allocator.free_count()
    }

    /// The pages backing `seq` in logical-block order.
    pub fn pages(
        &self,
        seq: SequenceId,
    ) -> &[PageId] {
        self.table.pages(seq)
    }

    /// Append a freshly allocated page to `seq` (its next logical block).
    /// Returns `None` if the pool is exhausted.
    pub fn append_page(
        &mut self,
        seq: SequenceId,
    ) -> Option<PageId> {
        let page = self.allocator.alloc()?;
        self.table.append_page(seq, page);
        Some(page)
    }

    /// Build the kernel `slot_mapping` for `seq` at the given token positions.
    pub fn slot_mapping(
        &self,
        seq: SequenceId,
        positions: &[usize],
    ) -> Vec<i64> {
        self.table.build_slot_mapping(seq, positions)
    }

    /// Make `child` share every page currently held by `parent`, bumping each
    /// page's reference count. Used for prefix sharing / sequence forks.
    ///
    /// # Panics
    /// If `child` already has pages registered.
    pub fn fork(
        &mut self,
        parent: SequenceId,
        child: SequenceId,
    ) {
        assert!(
            self.table.pages(child).is_empty(),
            "BlockManager: fork target {child} already has pages"
        );
        let shared: Vec<PageId> = self.table.pages(parent).to_vec();
        for &page in &shared {
            self.allocator.retain(page);
            self.table.append_page(child, page);
        }
    }

    /// Ensure the logical `block` of `seq` is uniquely owned before a write.
    /// If the page is shared (ref count > 1), allocate a fresh page, point the
    /// table at it, drop the shared reference, and return [`CowOutcome::Copied`]
    /// so the caller can copy the old page's bytes into the new one. Otherwise
    /// returns [`CowOutcome::AlreadyOwned`].
    ///
    /// # Panics
    /// If `block` is out of range for `seq`.
    /// # Errors-as-None
    /// Returns `None` only when a copy is required but the pool is exhausted.
    pub fn ensure_writable(
        &mut self,
        seq: SequenceId,
        block: usize,
    ) -> Option<CowOutcome> {
        let pages = self.table.pages(seq);
        let old = *pages
            .get(block)
            .unwrap_or_else(|| panic!("BlockManager: {seq} has no block {block}"));

        if self.allocator.ref_count(old) <= 1 {
            return Some(CowOutcome::AlreadyOwned(old));
        }

        let fresh = self.allocator.alloc()?;
        self.table.replace_page(seq, block, fresh);
        self.allocator.release(old);
        Some(CowOutcome::Copied {
            from: old,
            to: fresh,
        })
    }

    /// Release every page held by `seq` and forget it. Pages shared with other
    /// sequences survive until their last holder releases them.
    pub fn free_sequence(
        &mut self,
        seq: SequenceId,
    ) {
        for page in self.table.take_pages(seq) {
            self.allocator.release(page);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: u64) -> SequenceId {
        SequenceId::new(n)
    }

    #[test]
    fn append_consumes_free_pages() {
        let mut bm = BlockManager::new(4, 16);
        assert_eq!(bm.free_pages(), 4);
        let p = bm.append_page(seq(1)).unwrap();
        assert_eq!(bm.free_pages(), 3);
        assert_eq!(bm.pages(seq(1)), &[p]);
    }

    #[test]
    fn fork_shares_pages_without_new_allocation() {
        let mut bm = BlockManager::new(4, 16);
        bm.append_page(seq(1));
        bm.append_page(seq(1));
        let before = bm.free_pages();
        bm.fork(seq(1), seq(2));
        // Sharing must not consume physical pages.
        assert_eq!(bm.free_pages(), before);
        assert_eq!(bm.pages(seq(1)), bm.pages(seq(2)));
    }

    #[test]
    fn cow_copies_shared_block_and_keeps_unique_block() {
        let mut bm = BlockManager::new(4, 16);
        let p0 = bm.append_page(seq(1)).unwrap();
        bm.fork(seq(1), seq(2));

        // Block 0 is shared by seq1 and seq2 -> writing forces a copy.
        match bm.ensure_writable(seq(2), 0).unwrap() {
            CowOutcome::Copied { from, to } => {
                assert_eq!(from, p0);
                assert_ne!(to, p0);
                assert_eq!(bm.pages(seq(2)), &[to]);
                assert_eq!(bm.pages(seq(1)), &[p0]);
            },
            other => panic!("expected copy, got {other:?}"),
        }

        // Now seq2's block 0 is unique -> a second write needs no copy.
        let unique = bm.pages(seq(2))[0];
        assert_eq!(
            bm.ensure_writable(seq(2), 0).unwrap(),
            CowOutcome::AlreadyOwned(unique)
        );
    }

    #[test]
    fn free_sequence_keeps_pages_shared_by_others() {
        let mut bm = BlockManager::new(4, 16);
        bm.append_page(seq(1));
        bm.fork(seq(1), seq(2));
        let after_fork = bm.free_pages();
        bm.free_sequence(seq(1));
        // seq2 still holds the shared page, so nothing returns to the pool.
        assert_eq!(bm.free_pages(), after_fork);
        bm.free_sequence(seq(2));
        // Now the last holder is gone -> reclaimed.
        assert_eq!(bm.free_pages(), after_fork + 1);
    }
}
