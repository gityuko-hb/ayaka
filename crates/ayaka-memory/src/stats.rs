use core::sync::atomic::{AtomicUsize, Ordering};

use crate::{MemoryError, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryPurpose {
    Weights,
    KvSlab,
    Workspace,
    Pool,
    Transient,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub weights: usize,
    pub kv_slab: usize,
    pub workspace: usize,
    pub pool: usize,
    pub transient: usize,
}

impl MemorySnapshot {
    pub const fn total_tracked(self) -> usize {
        self.weights + self.kv_slab + self.workspace + self.pool + self.transient
    }
}

#[derive(Debug)]
pub struct MemoryLedger {
    weights: AtomicUsize,
    kv_slab: AtomicUsize,
    workspace: AtomicUsize,
    pool: AtomicUsize,
    transient: AtomicUsize,
}

impl MemoryLedger {
    pub const fn new() -> Self {
        Self {
            weights: AtomicUsize::new(0),
            kv_slab: AtomicUsize::new(0),
            workspace: AtomicUsize::new(0),
            pool: AtomicUsize::new(0),
            transient: AtomicUsize::new(0),
        }
    }

    pub fn register(&self, purpose: MemoryPurpose, bytes: usize) {
        self.counter(purpose).fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn deregister(&self, purpose: MemoryPurpose, bytes: usize) -> Result<()> {
        self.counter(purpose)
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| old.checked_sub(bytes))
            .map(|_| ())
            .map_err(|_| {
                MemoryError::invalid(format!(
                    "ledger deregister underflow for {purpose:?}: {bytes} bytes"
                ))
            })
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            weights: self.weights.load(Ordering::Relaxed),
            kv_slab: self.kv_slab.load(Ordering::Relaxed),
            workspace: self.workspace.load(Ordering::Relaxed),
            pool: self.pool.load(Ordering::Relaxed),
            transient: self.transient.load(Ordering::Relaxed),
        }
    }

    fn counter(&self, purpose: MemoryPurpose) -> &AtomicUsize {
        match purpose {
            MemoryPurpose::Weights => &self.weights,
            MemoryPurpose::KvSlab => &self.kv_slab,
            MemoryPurpose::Workspace => &self.workspace,
            MemoryPurpose::Pool => &self.pool,
            MemoryPurpose::Transient => &self.transient,
        }
    }
}

impl Default for MemoryLedger {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_LEDGER: MemoryLedger = MemoryLedger::new();

pub fn global_ledger() -> &'static MemoryLedger {
    &GLOBAL_LEDGER
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DriverMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MemoryStats {
    pub ledger: MemorySnapshot,
    pub driver: Option<DriverMemoryInfo>,
}

impl MemoryStats {
    pub fn untracked_used_bytes(self) -> Option<usize> {
        let driver = self.driver?;
        Some(driver.total_bytes.saturating_sub(driver.free_bytes).saturating_sub(
            self.ledger.total_tracked(),
        ))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct KvBudget {
    pub kv_budget_bytes: usize,
    pub block_bytes: usize,
    pub num_blocks: usize,
}

impl KvBudget {
    pub fn from_free_memory(
        free_bytes: usize,
        total_bytes: usize,
        block_bytes: usize,
        safety_margin_fraction: f64,
    ) -> Result<Self> {
        if block_bytes == 0 {
            return Err(MemoryError::invalid("block_bytes must be non-zero"));
        }
        if !(0.0..=1.0).contains(&safety_margin_fraction) {
            return Err(MemoryError::invalid(format!(
                "safety_margin_fraction must be in [0, 1], got {safety_margin_fraction}"
            )));
        }
        let safety_margin = (total_bytes as f64 * safety_margin_fraction) as usize;
        let kv_budget_bytes = free_bytes.saturating_sub(safety_margin);
        Ok(Self {
            kv_budget_bytes,
            block_bytes,
            num_blocks: kv_budget_bytes / block_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_tracks_purpose_totals() {
        let ledger = MemoryLedger::new();
        ledger.register(MemoryPurpose::KvSlab, 1024);
        ledger.register(MemoryPurpose::Workspace, 256);
        ledger.deregister(MemoryPurpose::KvSlab, 24).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.kv_slab, 1000);
        assert_eq!(snapshot.workspace, 256);
        assert_eq!(snapshot.total_tracked(), 1256);
    }

    #[test]
    fn ledger_rejects_underflow() {
        let ledger = MemoryLedger::new();
        assert!(ledger.deregister(MemoryPurpose::Pool, 1).is_err());
    }

    #[test]
    fn budget_uses_safety_margin() {
        let budget = KvBudget::from_free_memory(10_000, 20_000, 1024, 0.05).unwrap();
        assert_eq!(budget.kv_budget_bytes, 9_000);
        assert_eq!(budget.num_blocks, 8);
    }
}
