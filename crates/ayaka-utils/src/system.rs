use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{CpuExt, DiskExt, NetworkExt, PidExt, ProcessExt, System, SystemExt};

#[derive(Debug, Clone)]
pub struct SystemInfo {
    // --- CPU ---
    pub cpu_brand: String, // "Intel(R) Core(TM) i9-13900K CPU @ 3.00GHz"
    pub cpu_physical_cores: usize, // physical cores
    pub cpu_logical_cores: usize, // logical cores
    pub cpu_base_freq_mhz: u64, // base clock MHz
    pub cpu_arch: CpuArch,

    // --- RAM ---
    pub total_ram_bytes: usize,
    pub swap_total_bytes: usize,

    // --- OS ---
    pub os_name: String,    // "Linux", "macOS", "Windows"
    pub os_version: String, // "Ubuntu 22.04" / "macOS 14.2"
    pub kernel_version: String,
    pub hostname: String,

    // --- Process ---
    pub pid: u32,
    pub exe_path: String,
    pub start_time: u64, // UNIX timestamp

    // --- Engine capabilities ---
    pub supports_avx512: bool,   // x86 AVX-512
    pub supports_amx: bool,      // x86 AMX (Intel Sapphire Rapids+)
    pub supports_bf16_cpu: bool, // BF16 native support on CPU
    pub page_size_bytes: usize,  // OS memory page size
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86_64,  // AMD64
    Aarch64, // ARM64
    Riscv64, // RISC-V 64-bit
    Unknown,
}

impl CpuArch {
    pub fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::Aarch64
        }
        #[cfg(target_arch = "riscv64")]
        {
            Self::Riscv64
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            Self::Unknown
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Riscv64 => "riscv64",
            Self::Unknown => "unknown",
        }
    }
}

static SYSTEM_INFO: std::sync::OnceLock<SystemInfo> = std::sync::OnceLock::new();

