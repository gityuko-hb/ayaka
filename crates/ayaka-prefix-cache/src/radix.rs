//! Radix (PATRICIA) prefix cache for KV reuse across requests.
//!
//! Maps token prefixes to the physical KV pages already computed for them, so a
//! new request sharing a prefix can skip prefilling it. Edges are
//! length-compressed; a divergence splits the shared edge. Nodes carry a
//! reference count (pin a matched prefix while a sequence is using it) and a
//! last-access clock for leaf-LRU eviction. Pure Rust, arena-backed, no FFI.

use std::collections::HashMap;

use ayaka_core::id::{PageId, TokenId};

const NULL: u32 = u32::MAX;

#[derive(Debug)]
struct Node {
    /// Length-compressed edge tokens from parent to this node.
    tokens: Vec<TokenId>,
    /// KV pages covering this edge's tokens (1:1 with `tokens`).
    pages: Vec<PageId>,
    /// Children keyed by the first token of their edge.
    children: HashMap<TokenId, u32>,
    parent: u32,
    ref_count: u32,
    last_access: u64,
    dead: bool,
}

/// Result of a longest-prefix lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Number of leading tokens matched.
    pub length: usize,
    /// Cached pages for the matched prefix, in order.
    pub pages: Vec<PageId>,
    /// Deepest node touched (pass to [`RadixCache::acquire`] to pin it).
    pub node: u32,
}

/// A radix prefix cache over token sequences.
#[derive(Debug)]
pub struct RadixCache {
    nodes: Vec<Node>,
    clock: u64,
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixCache {
    /// An empty cache (just the root).
    pub fn new() -> Self {
        let root = Node {
            tokens: Vec::new(),
            pages: Vec::new(),
            children: HashMap::new(),
            parent: NULL,
            ref_count: 0,
            last_access: 0,
            dead: false,
        };
        Self {
            nodes: vec![root],
            clock: 0,
        }
    }

    fn alloc(
        &mut self,
        tokens: Vec<TokenId>,
        pages: Vec<PageId>,
        parent: u32,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            tokens,
            pages,
            children: HashMap::new(),
            parent,
            ref_count: 0,
            last_access: self.clock,
            dead: false,
        });
        id
    }

    fn touch(
        &mut self,
        id: u32,
    ) {
        self.clock += 1;
        self.nodes[id as usize].last_access = self.clock;
    }

    /// Insert `tokens` and their `pages` (equal length), splitting shared edges
    /// as needed. Returns the node that terminates the inserted prefix.
    pub fn insert(
        &mut self,
        tokens: &[TokenId],
        pages: &[PageId],
    ) -> u32 {
        assert_eq!(tokens.len(), pages.len(), "tokens/pages length mismatch");
        let mut cur = 0u32;
        let mut i = 0usize;
        while i < tokens.len() {
            let first = tokens[i];
            match self.nodes[cur as usize]
                .children
                .get(&first)
                .copied()
            {
                None => {
                    let id = self.alloc(tokens[i..].to_vec(), pages[i..].to_vec(), cur);
                    self.nodes[cur as usize]
                        .children
                        .insert(first, id);
                    self.touch(id);
                    return id;
                },
                Some(child) => {
                    let lcp = common_prefix_len(&tokens[i..], &self.nodes[child as usize].tokens);
                    let edge_len = self.nodes[child as usize].tokens.len();
                    if lcp == edge_len {
                        i += lcp;
                        cur = child;
                        self.touch(cur);
                        if i == tokens.len() {
                            return cur;
                        }
                    } else {
                        let mid = self.split(child, lcp);
                        i += lcp;
                        cur = mid;
                        self.touch(cur);
                        if i == tokens.len() {
                            return cur;
                        }
                        let id = self.alloc(tokens[i..].to_vec(), pages[i..].to_vec(), mid);
                        let first2 = tokens[i];
                        self.nodes[mid as usize]
                            .children
                            .insert(first2, id);
                        self.touch(id);
                        return id;
                    }
                },
            }
        }
        cur
    }

    /// Split `child`'s edge at `lcp`, inserting an intermediate node holding the
    /// shared head; returns the intermediate node id.
    fn split(
        &mut self,
        child: u32,
        lcp: usize,
    ) -> u32 {
        let parent = self.nodes[child as usize].parent;
        let head_tokens = self.nodes[child as usize].tokens[..lcp].to_vec();
        let head_pages = self.nodes[child as usize].pages[..lcp].to_vec();
        let key = head_tokens[0];

        self.nodes[child as usize].tokens.drain(..lcp);
        self.nodes[child as usize].pages.drain(..lcp);
        let child_first = self.nodes[child as usize].tokens[0];

        let mid = self.alloc(head_tokens, head_pages, parent);
        self.nodes[mid as usize].last_access = self.nodes[child as usize].last_access;
        self.nodes[mid as usize]
            .children
            .insert(child_first, child);
        self.nodes[child as usize].parent = mid;
        self.nodes[parent as usize]
            .children
            .insert(key, mid);
        mid
    }

    /// Longest-prefix match for `tokens`, accumulating the cached pages.
    pub fn match_prefix(
        &mut self,
        tokens: &[TokenId],
    ) -> PrefixMatch {
        let mut cur = 0u32;
        let mut i = 0usize;
        let mut pages = Vec::new();
        while i < tokens.len() {
            let first = tokens[i];
            let Some(child) = self.nodes[cur as usize]
                .children
                .get(&first)
                .copied()
            else {
                break;
            };
            let lcp = common_prefix_len(&tokens[i..], &self.nodes[child as usize].tokens);
            pages.extend_from_slice(&self.nodes[child as usize].pages[..lcp]);
            i += lcp;
            cur = child;
            self.touch(cur);
            if lcp < self.nodes[child as usize].tokens.len() {
                break; // partial edge match
            }
        }
        PrefixMatch {
            length: i,
            pages,
            node: cur,
        }
    }

    /// Pin `node` and its ancestors so they survive eviction.
    pub fn acquire(
        &mut self,
        node: u32,
    ) {
        let mut c = node;
        loop {
            self.nodes[c as usize].ref_count += 1;
            let p = self.nodes[c as usize].parent;
            if p == NULL {
                break;
            }
            c = p;
        }
    }

    /// Release a previous [`acquire`](Self::acquire).
    pub fn release(
        &mut self,
        node: u32,
    ) {
        let mut c = node;
        loop {
            let rc = &mut self.nodes[c as usize].ref_count;
            *rc = rc.saturating_sub(1);
            let p = self.nodes[c as usize].parent;
            if p == NULL {
                break;
            }
            c = p;
        }
    }

    /// Evict unreferenced leaves in LRU order until at least `max_pages` pages
    /// are reclaimed (or no evictable leaf remains). Returns the freed pages.
    pub fn evict(
        &mut self,
        max_pages: usize,
    ) -> Vec<PageId> {
        let mut leaves: Vec<u32> = (1..self.nodes.len() as u32)
            .filter(|&id| {
                let n = &self.nodes[id as usize];
                !n.dead && n.ref_count == 0 && n.children.is_empty()
            })
            .collect();
        leaves.sort_by_key(|&id| self.nodes[id as usize].last_access);

        let mut freed = Vec::new();
        for id in leaves {
            if freed.len() >= max_pages {
                break;
            }
            let parent = self.nodes[id as usize].parent;
            let key = self.nodes[id as usize].tokens[0];
            self.nodes[parent as usize].children.remove(&key);
            let pages: Vec<PageId> = self.nodes[id as usize].pages.drain(..).collect();
            freed.extend(pages);
            self.nodes[id as usize].dead = true;
        }
        freed
    }

    /// Total cached pages across live nodes.
    pub fn cached_pages(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| !n.dead)
            .map(|n| n.pages.len())
            .sum()
    }
}

fn common_prefix_len(
    a: &[TokenId],
    b: &[TokenId],
) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(x, y)| x == y)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(ids: &[u32]) -> Vec<PageId> {
        ids.iter().map(|&i| PageId::new(i)).collect()
    }

    #[test]
    fn exact_then_prefix_match() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &pages(&[10, 11, 12]));
        let m = c.match_prefix(&[1, 2, 3]);
        assert_eq!(m.length, 3);
        assert_eq!(m.pages, pages(&[10, 11, 12]));

        let p = c.match_prefix(&[1, 2]);
        assert_eq!(p.length, 2);
        assert_eq!(p.pages, pages(&[10, 11]));

        assert_eq!(c.match_prefix(&[9]).length, 0);
    }

    #[test]
    fn shared_prefix_splits_and_reuses() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2, 3], &pages(&[10, 11, 12]));
        c.insert(&[1, 2, 4], &pages(&[10, 11, 13]));

        // Both full sequences resolve.
        assert_eq!(c.match_prefix(&[1, 2, 3]).pages, pages(&[10, 11, 12]));
        assert_eq!(c.match_prefix(&[1, 2, 4]).pages, pages(&[10, 11, 13]));
        // Shared prefix only.
        let s = c.match_prefix(&[1, 2, 9]);
        assert_eq!(s.length, 2);
        assert_eq!(s.pages, pages(&[10, 11]));
        // Partial match inside the [1,2] edge.
        let part = c.match_prefix(&[1, 5]);
        assert_eq!(part.length, 1);
        assert_eq!(part.pages, pages(&[10]));
    }

    #[test]
    fn divergent_first_token_makes_separate_branches() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2], &pages(&[10, 11]));
        c.insert(&[3, 4], &pages(&[20, 21]));
        assert_eq!(c.match_prefix(&[1, 2]).pages, pages(&[10, 11]));
        assert_eq!(c.match_prefix(&[3, 4]).pages, pages(&[20, 21]));
        assert_eq!(c.cached_pages(), 4);
    }

    #[test]
    fn evicts_lru_leaf_and_frees_pages() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2], &pages(&[10, 11])); // older
        c.insert(&[3, 4], &pages(&[20, 21])); // newer

        let freed = c.evict(1);
        assert_eq!(freed, pages(&[10, 11])); // LRU leaf evicted
        assert_eq!(c.match_prefix(&[1, 2]).length, 0); // gone
        assert_eq!(c.match_prefix(&[3, 4]).length, 2); // survives
    }

    #[test]
    fn pinned_leaf_is_not_evicted() {
        let mut c = RadixCache::new();
        let a = c.insert(&[1, 2], &pages(&[10, 11])); // older, will be pinned
        c.insert(&[3, 4], &pages(&[20, 21])); // newer
        c.acquire(a);

        let freed = c.evict(10); // try to free everything
        // Pinned [1,2] survives; only [3,4] is reclaimed.
        assert_eq!(freed, pages(&[20, 21]));
        assert_eq!(c.match_prefix(&[1, 2]).length, 2);

        c.release(a);
        let freed2 = c.evict(10);
        assert_eq!(freed2, pages(&[10, 11]));
    }

    #[test]
    fn extend_existing_prefix() {
        let mut c = RadixCache::new();
        c.insert(&[1, 2], &pages(&[10, 11]));
        c.insert(&[1, 2, 3, 4], &pages(&[10, 11, 12, 13]));
        let m = c.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(m.length, 4);
        assert_eq!(m.pages, pages(&[10, 11, 12, 13]));
    }
}
