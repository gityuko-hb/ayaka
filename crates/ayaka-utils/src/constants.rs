//! Centralized constants for the inference engine.
//!
//! All magic numbers, default values, environment variable names,
//! validation bounds, and byte conversion factors live here
//! to avoid repetition and ensure consistency across crates.

// ═══════════════════════════════════════════════
// Byte / Size Conversions
// ═══════════════════════════════════════════════

pub const BYTES_PER_KIB: u64 = 1024;
pub const BYTES_PER_MIB: u64 = 1048576;
pub const BYTES_PER_GIB: u64 = 1073741824;

// ═══════════════════════════════════════════════
// Lock / Spin / Timeout
// ═══════════════════════════════════════════════

/// Spin iterations before warning about a potential deadlock (~1 ms on modern HW).
pub const LOCK_SPIN_WARN_THRESHOLD: u32 = 1_000;
/// Default timeout for lock acquisition (ms).
pub const LOCK_TIMEOUT_DEFAULT_MS: u64 = 5_000;

// ═══════════════════════════════════════════════
// Progress Bar / UI
// ═══════════════════════════════════════════════

pub const PROGRESS_BAR_WIDTH: u32 = 40;
pub const PROGRESS_TICK_MS: u64 = 100;
pub const PROGRESS_TICK_CHARS: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"];

// ═══════════════════════════════════════════════
// Default Configuration Values
// ═══════════════════════════════════════════════

pub const DEFAULT_LOG_LEVEL: &str = "info";
pub const DEFAULT_GPU_MEMORY_FRACTION: f64 = 0.90;
pub const DEFAULT_IGPU_MEMORY_FRACTION: f64 = 0.75;
pub const DEFAULT_KV_BLOCK_SIZE: usize = 16;
pub const DEFAULT_METRICS_INTERVAL_SECS: u64 = 30;
pub const DEFAULT_RESERVED_FRACTION: f64 = 0.05;
pub const DEFAULT_CPU_COUNT_FALLBACK: usize = 4;

// ═══════════════════════════════════════════════
// Validation Bounds
// ═══════════════════════════════════════════════

pub const GPU_MEMORY_FRACTION_MIN: f64 = 0.1;
pub const GPU_MEMORY_FRACTION_MAX: f64 = 0.98;
pub const IGPU_MEMORY_FRACTION_MIN: f64 = 0.1;
pub const IGPU_MEMORY_FRACTION_MAX: f64 = 0.95;
pub const KV_BLOCK_SIZE_MIN: usize = 8;
/// Warn if load_threads > cpu_count × LOAD_THREADS_THRASHING_RATIO.
pub const LOAD_THREADS_THRASHING_RATIO: usize = 4;

// ═══════════════════════════════════════════════
// Environment Variable Names
// ═══════════════════════════════════════════════

pub const ENV_LOG_LEVEL: &str = "ENGINE_LOG_LEVEL";
pub const ENV_LOG_JSON: &str = "ENGINE_LOG_JSON";
pub const ENV_DEBUG: &str = "ENGINE_DEBUG";
pub const ENV_GPU_MEMORY_FRACTION: &str = "ENGINE_GPU_MEMORY_FRACTION";
pub const ENV_IGPU_MEMORY_FRACTION: &str = "ENGINE_IGPU_MEMORY_FRACTION";
pub const ENV_CPU_SWAP_GB: &str = "ENGINE_CPU_SWAP_GB";
pub const ENV_NO_MMAP: &str = "ENGINE_NO_MMAP";
pub const ENV_LOAD_THREADS: &str = "ENGINE_LOAD_THREADS";
pub const ENV_MAX_BATCH_SIZE: &str = "ENGINE_MAX_BATCH_SIZE";
pub const ENV_MAX_SEQ_LEN: &str = "ENGINE_MAX_SEQ_LEN";
pub const ENV_SEED: &str = "ENGINE_SEED";
pub const ENV_KV_BLOCK_SIZE: &str = "ENGINE_KV_BLOCK_SIZE";
pub const ENV_PROFILE: &str = "ENGINE_PROFILE";
pub const ENV_METRICS_INTERVAL: &str = "ENGINE_METRICS_INTERVAL";
pub const ENV_LOG_PREFIX: &str = "ENGINE_LOG_";
