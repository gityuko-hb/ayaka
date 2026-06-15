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

/// Free-list allocator over a fixed set of physical pages.
#[derive(Debug)]
pub struct PageAllocator {
    /// `next[i]`: the next free page after page `i`, or [`NULL`] at the tail.
    /// For an allocated page the entry is unused.
    next: Vec<u32>,
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

    /// Take a free page, or `None` if the pool is exhausted.
    pub fn alloc(&mut self) -> Option<PageId> {
        if self.head == NULL {
            return None;
        }
        let page = self.head;
        self.head = self.next[page as usize];
        self.free_count -= 1;
        Some(PageId::new(page))
    }

    /// Return a previously allocated page to the pool.
    ///
    /// # Panics
    /// If `page` is out of range, or (in debug builds) freeing it would push
    /// the free count past capacity — a sign of a double free.
    pub fn free(
        &mut self,
        page: PageId,
    ) {
        let idx = page.raw();
        assert!(
            (idx as usize) < self.next.len(),
            "PageAllocator: freed page {idx} out of range (capacity {})",
            self.next.len()
        );
        debug_assert!(
            (self.free_count as usize) < self.next.len(),
            "PageAllocator: free of {idx} exceeds capacity (double free?)"
        );
        self.next[idx as usize] = self.head;
        self.head = idx;
        self.free_count += 1;
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
}
