//! Physical-page allocator for the paged KV cache.
//!
//! Pure logical bookkeeping over `num_blocks` physical pages — it owns no
//! device memory, just decides which page indices are free. Backed by an
//! intrusive free-list (`next[i]` is the next free page after `i`, or
//! [`NULL`]), so `alloc`/`free` are O(1) and allocate nothing after `new`.

use ayaka_core::id::PageId;

/// Sentinel for "no next page"; also bounds the page count (a page index can
/// never equal it).
const NULL: u32 = u32::MAX;

/// Free-list allocator over a fixed set of physical pages, with per-page
/// reference counts so a page can be shared across sequences (prefix reuse /
/// fork). A page returns to the free list only when its count reaches zero.
#[derive(Debug)]
pub struct PageAllocator {
    /// `next[i]`: the next free page after page `i`, or [`NULL`] at the tail.
    /// For an allocated page the entry is unused.
    next: Vec<u32>,
    /// `ref_count[i]`: number of holders of page `i`; `0` means free.
    ref_count: Vec<u32>,
    /// Head of the free list, or [`NULL`] when no page is free.
    head: u32,
    /// Number of currently free pages (invariant: equals the free-list length).
    free_count: u32,
}

impl PageAllocator {
    /// Allocator over `num_blocks` pages, all initially free.
    ///
    /// # Panics
    /// If `num_blocks >= u32::MAX` (the largest value is reserved as the
    /// free-list sentinel).
    pub fn new(num_blocks: usize) -> Self {
        assert!(
            num_blocks < NULL as usize,
            "PageAllocator: num_blocks {num_blocks} must be < u32::MAX"
        );
        let n = num_blocks as u32;
        let next: Vec<u32> = (0..n)
            .map(|i| {
                if i + 1 < n {
                    i + 1
                } else {
                    NULL
                }
            })
            .collect();
        let head = if n == 0 {
            NULL
        } else {
            0
        };
        Self {
            next,
            ref_count: vec![0; num_blocks],
            head,
            free_count: n,
        }
    }

    /// Total number of pages this allocator manages.
    pub fn capacity(&self) -> usize {
        self.next.len()
    }

    /// Number of pages currently free.
    pub fn free_count(&self) -> usize {
        self.free_count as usize
    }

    /// Take a free page with reference count 1, or `None` if the pool is
    /// exhausted.
    pub fn alloc(&mut self) -> Option<PageId> {
        if self.head == NULL {
            return None;
        }
        let page = self.head;
        self.head = self.next[page as usize];
        self.free_count -= 1;
        self.ref_count[page as usize] = 1;
        Some(PageId::new(page))
    }

    /// Add a reference to an already-allocated page (shared ownership, e.g.
    /// fork / prefix reuse). Returns the new reference count.
    ///
    /// # Panics
    /// If `page` is out of range or currently free (count 0).
    pub fn retain(
        &mut self,
        page: PageId,
    ) -> u32 {
        let idx = self.checked_index(page);
        assert!(
            self.ref_count[idx] > 0,
            "PageAllocator: retain of free page {}",
            page.raw()
        );
        self.ref_count[idx] += 1;
        self.ref_count[idx]
    }

    /// Drop one reference to `page`. The page returns to the free list only
    /// when its count reaches zero. Returns the remaining reference count
    /// (`0` means it was reclaimed).
    ///
    /// # Panics
    /// If `page` is out of range or already free (double free).
    pub fn release(
        &mut self,
        page: PageId,
    ) -> u32 {
        let idx = self.checked_index(page);
        assert!(
            self.ref_count[idx] > 0,
            "PageAllocator: release of free page {} (double free?)",
            page.raw()
        );
        self.ref_count[idx] -= 1;
        if self.ref_count[idx] == 0 {
            self.next[idx] = self.head;
            self.head = idx as u32;
            self.free_count += 1;
        }
        self.ref_count[idx]
    }

    /// Current reference count of `page` (`0` if free).
    pub fn ref_count(
        &self,
        page: PageId,
    ) -> u32 {
        self.ref_count[self.checked_index(page)]
    }

    /// Backwards-compatible alias for [`PageAllocator::release`]; drops one
    /// reference.
    pub fn free(
        &mut self,
        page: PageId,
    ) {
        let _ = self.release(page);
    }

    fn checked_index(
        &self,
        page: PageId,
    ) -> usize {
        let idx = page.raw() as usize;
        assert!(
            idx < self.next.len(),
            "PageAllocator: page {} out of range (capacity {})",
            page.raw(),
            self.next.len()
        );
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_starts_all_free() {
        let a = PageAllocator::new(4);
        assert_eq!(a.capacity(), 4);
        assert_eq!(a.free_count(), 4);
    }

    #[test]
    fn alloc_returns_distinct_pages_then_exhausts() {
        let mut a = PageAllocator::new(3);
        let mut seen = HashSet::new();
        for _ in 0..3 {
            let p = a.alloc().expect("page available");
            assert!(seen.insert(p.raw()), "page {p} handed out twice");
        }
        assert_eq!(a.free_count(), 0);
        assert!(a.alloc().is_none(), "exhausted allocator must return None");
    }

    #[test]
    fn free_makes_page_available_again() {
        let mut a = PageAllocator::new(2);
        let p0 = a.alloc().unwrap();
        let _p1 = a.alloc().unwrap();
        assert!(a.alloc().is_none());
        a.free(p0);
        assert_eq!(a.free_count(), 1);
        let p = a.alloc().expect("freed page is reusable");
        assert_eq!(p, p0);
    }

    #[test]
    fn free_all_then_realloc_all() {
        let mut a = PageAllocator::new(4);
        let pages: Vec<_> = (0..4).map(|_| a.alloc().unwrap()).collect();
        for p in pages {
            a.free(p);
        }
        assert_eq!(a.free_count(), 4);
        let again: HashSet<u32> = (0..4).map(|_| a.alloc().unwrap().raw()).collect();
        assert_eq!(again.len(), 4);
    }

    #[test]
    fn empty_allocator_yields_none() {
        let mut a = PageAllocator::new(0);
        assert_eq!(a.capacity(), 0);
        assert!(a.alloc().is_none());
    }

    #[test]
    fn alloc_sets_ref_count_one() {
        let mut a = PageAllocator::new(2);
        let p = a.alloc().unwrap();
        assert_eq!(a.ref_count(p), 1);
    }

    #[test]
    fn shared_page_reclaimed_only_at_zero() {
        let mut a = PageAllocator::new(1); // single page -> easy exhaustion check
        let p = a.alloc().unwrap();
        assert_eq!(a.retain(p), 2);
        // Still held: the pool is exhausted and the page is not free.
        assert!(a.alloc().is_none());
        assert_eq!(a.release(p), 1); // one holder left, not reclaimed
        assert_eq!(a.free_count(), 0);
        assert!(a.alloc().is_none());
        assert_eq!(a.release(p), 0); // last holder -> reclaimed
        assert_eq!(a.free_count(), 1);
        assert_eq!(a.alloc().unwrap(), p);
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn release_of_free_page_panics() {
        let mut a = PageAllocator::new(1);
        let p = a.alloc().unwrap();
        a.release(p);
        a.release(p); // already free
    }
}
