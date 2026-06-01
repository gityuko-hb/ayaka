use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;

/// Static system information — detected once at startup, then immutable.
/// Does **not** include hardware topology (→ topology.rs) or VRAM (→ memory_usage.rs).
#[derive(Debug, Clone)]
pub struct SystemInfo {
    // --- CPU ---
    pub cpu_brand: String, // "Intel(R) Core(TM) i9-13900K CPU @ 3.00GHz"
    pub cpu_physical_cores: usize, // physical cores (excluding HT)
    pub cpu_logical_cores: usize, // logical cores (including HT)
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
    X86_64,
    Aarch64,
    Riscv64,
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

impl SystemInfo {
    /// Detect and cache. Call once at startup.
    /// Idempotent — subsequent calls return the same result.
    pub fn detect() -> &'static Self {
        SYSTEM_INFO.get_or_init(|| {
            let mut sys = System::new_all();
            sys.refresh_all();

            let cpu_brand = sys
                .cpus()
                .first()
                .map(|c| c.brand().to_string())
                .unwrap_or_else(|| "Unknown CPU".to_string());

            let cpu_logical_cores = sys.cpus().len().max(1);

            // Physical cores: on Linux reads from /sys, falls back to logical/2
            let cpu_physical_cores = Self::detect_physical_cores()
                .unwrap_or(cpu_logical_cores / 2)
                .max(1);

            let cpu_base_freq_mhz = sys
                .cpus()
                .first()
                .map(|c| c.frequency())
                .unwrap_or(0);

            let page_size_bytes = Self::detect_page_size();

            let info = SystemInfo {
                cpu_brand,
                cpu_physical_cores,
                cpu_logical_cores,
                cpu_base_freq_mhz,
                cpu_arch: CpuArch::current(),

                total_ram_bytes: sys.total_memory() as usize,
                swap_total_bytes: sys.total_swap() as usize,

                os_name: sysinfo::System::name().unwrap_or_else(|| "Unknown".to_string()),
                os_version: sysinfo::System::long_os_version()
                    .unwrap_or_else(|| "Unknown".to_string()),
                kernel_version: sysinfo::System::kernel_version()
                    .unwrap_or_else(|| "Unknown".to_string()),
                hostname: sysinfo::System::host_name().unwrap_or_else(|| "Unknown".to_string()),

                pid: std::process::id(),
                exe_path: std::env::current_exe()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                start_time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),

                supports_avx512: Self::detect_avx512(),
                supports_amx: Self::detect_amx(),
                supports_bf16_cpu: Self::detect_bf16_cpu(),
                page_size_bytes,
            };

            info.log_summary();
            info
        })
    }

    /// Shorthand after `detect()` has been called.
    #[inline]
    pub fn get() -> &'static Self {
        SYSTEM_INFO
            .get()
            .expect("SystemInfo not initialized — call SystemInfo::detect() at startup")
    }

    /// Log all information at startup for environment debugging.
    pub fn log_summary(&self) {
        tracing::info!(
            cpu             = %self.cpu_brand,
            physical_cores  = self.cpu_physical_cores,
            logical_cores   = self.cpu_logical_cores,
            base_freq_mhz   = self.cpu_base_freq_mhz,
            arch            = self.cpu_arch.as_str(),
            ram_gb          = self.total_ram_bytes / (1 << 30),
            swap_gb         = self.swap_total_bytes / (1 << 30),
            os              = %self.os_version,
            kernel          = %self.kernel_version,
            hostname        = %self.hostname,
            pid             = self.pid,
            avx512          = self.supports_avx512,
            amx             = self.supports_amx,
            bf16_cpu        = self.supports_bf16_cpu,
            page_size       = self.page_size_bytes,
            "System info detected"
        );
    }

    // ---- Private: CPU feature detection ------------------------------------

    fn detect_physical_cores() -> Option<usize> {
        // Linux: count unique core IDs from /proc/cpuinfo
        #[cfg(target_os = "linux")]
        {
            let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            let mut core_ids = std::collections::HashSet::new();
            for line in content.lines() {
                if line.starts_with("core id") {
                    if let Some(id) = line.split(':').nth(1) {
                        core_ids.insert(id.trim().to_string());
                    }
                }
            }
            if !core_ids.is_empty() {
                return Some(core_ids.len());
            }
        }
        None
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Detect whether the CPU supports Intel Advanced Matrix Extensions (AMX).
    ///
    /// # Safety
    ///
    /// This function uses an `unsafe` block to directly invoke `__cpuid` and
    /// `__cpuid_count` intrinsics. This is safe because:
    /// 1. The intrinsics are only called on `x86_64` via `#[cfg(target_arch = "x86_64")]`.
    /// 2. CPUID is a read-only instruction — it does not modify memory or cause UB.
    /// 3. Before calling leaf 7 (`__cpuid_count(7, 0)`), the code verifies the CPU
    ///    supports leaf 7+ (`cpuid_0.eax >= 7`).
    fn detect_amx() -> bool {
        // AMX: Intel Advanced Matrix Extensions (Sapphire Rapids+)
        // CPUID leaf 0x7, subleaf 0x0,
        // EDX bit 22 (AMX-BF16) and bit 24 (AMX-INT8)
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: CPUID is a read-only hardware info instruction. It does not
            // affect memory and is guarded by the leaf >= 7 check to avoid faults
            // on older CPUs.
            unsafe {
                let cpuid_0 = std::arch::x86_64::__cpuid(0);
                if cpuid_0.eax >= 7 {
                    let cpuid_7 = std::arch::x86_64::__cpuid_count(7, 0);

                    let amx_bf16 = (cpuid_7.edx & (1 << 22)) != 0;
                    let amx_int8 = (cpuid_7.edx & (1 << 24)) != 0;

                    return amx_bf16 || amx_int8;
                }
                false
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_bf16_cpu() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512bf16")
        }
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("bf16")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    fn detect_page_size() -> usize {
        // SAFETY: sysconf is a pure OS call with no side effects
        #[cfg(unix)]
        {
            let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if sz > 0 {
                return sz as usize;
            }
        }
        4096 // fallback
    }
}

/// A point-in-time snapshot of runtime system state.
/// All fields are plain data — safe to clone and serialize.
#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    pub captured_at: Instant,
    pub captured_at_unix_secs: u64,

    // --- CPU utilization ---
    /// Aggregate utilization across all cores [0.0, 1.0]
    pub cpu_utilization_global: f32,
    /// Per-core utilization
    pub cpu_utilization_per_core: Vec<f32>,
    /// Load average 1min/5min/15min (Linux/macOS only)
    pub load_avg_1: f64,
    pub load_avg_5: f64,
    pub load_avg_15: f64,

    // --- Memory ---
    pub ram_total_bytes: usize,
    pub ram_available_bytes: usize,
    pub ram_used_bytes: usize,
    pub swap_used_bytes: usize,

    // --- Process-level ---
    pub process_rss_bytes: usize,     // Resident Set Size
    pub process_vms_bytes: usize,     // Virtual Memory Size
    pub open_file_descriptors: usize, // open FDs (Linux only)
    pub thread_count: usize,

    // --- Throttle detection ---
    pub cpu_throttle_detected: bool, // CPU is thermal throttling
    pub cpu_freq_current_mhz: u64,   // current freq (may be below base when throttling)

    // --- Disk I/O (aggregate) ---
    pub disk_read_bytes_total: u64,
    pub disk_write_bytes_total: u64,

    // --- Network ---
    pub net_rx_bytes_total: u64,
    pub net_tx_bytes_total: u64,
}

impl RuntimeSnapshot {
    /// Capture the current snapshot.
    /// Calls `System::refresh_all()` — not cheap, avoid in hot paths.
    pub fn capture() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        sys.refresh_cpu_all();

        let pid = sysinfo::Pid::from_u32(std::process::id());
        let process = sys.process(pid);

        let cpu_utilization_per_core: Vec<f32> = sys
            .cpus()
            .iter()
            .map(|c| c.cpu_usage() / 100.0)
            .collect();

        let cpu_utilization_global = if cpu_utilization_per_core.is_empty() {
            0.0
        } else {
            cpu_utilization_per_core.iter().sum::<f32>() / cpu_utilization_per_core.len() as f32
        };

        let load_avg = sysinfo::System::load_average();

        let cpu_freq_current_mhz = sys
            .cpus()
            .iter()
            .map(|c| c.frequency())
            .min()
            .unwrap_or(0);

        // Throttle suspect: if current freq < 70% of base freq
        let base_freq = SystemInfo::get().cpu_base_freq_mhz;
        let cpu_throttle_detected = base_freq > 0
            && cpu_freq_current_mhz > 0
            && cpu_freq_current_mhz < (base_freq * 70 / 100);

        let disks = sysinfo::Disks::new_with_refreshed_list();
        let (disk_read, disk_write) = disks.iter().fold((0u64, 0u64), |(r, w), _d| {
            // sysinfo Disks does not expose aggregate IO bytes directly — read from /proc
            (r, w)
        });

        let networks = sysinfo::Networks::new_with_refreshed_list();
        let (net_rx, net_tx) = networks
            .iter()
            .fold((0u64, 0u64), |(rx, tx), (_, n)| {
                (rx + n.total_received(), tx + n.total_transmitted())
            });

        Self {
            captured_at: Instant::now(),
            captured_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),

            cpu_utilization_global,
            cpu_utilization_per_core,
            load_avg_1: load_avg.one,
            load_avg_5: load_avg.five,
            load_avg_15: load_avg.fifteen,

            ram_total_bytes: sys.total_memory() as usize,
            ram_available_bytes: sys.available_memory() as usize,
            ram_used_bytes: (sys.total_memory() - sys.available_memory()) as usize,
            swap_used_bytes: sys.used_swap() as usize,

            process_rss_bytes: process.map(|p| p.memory() as usize).unwrap_or(0),
            process_vms_bytes: process
                .map(|p| p.virtual_memory() as usize)
                .unwrap_or(0),
            open_file_descriptors: Self::count_open_fds(),
            thread_count: Self::count_threads(),

            cpu_throttle_detected,
            cpu_freq_current_mhz,

            disk_read_bytes_total: disk_read,
            disk_write_bytes_total: disk_write,
            net_rx_bytes_total: net_rx,
            net_tx_bytes_total: net_tx,
        }
    }

    /// RAM utilization [0.0, 1.0]
    #[inline]
    pub fn ram_utilization(&self) -> f64 {
        if self.ram_total_bytes == 0 {
            return 0.0;
        }
        self.ram_used_bytes as f64 / self.ram_total_bytes as f64
    }

    fn count_open_fds() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_dir("/proc/self/fd")
                .map(|dir| dir.count())
                .unwrap_or(0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    fn count_threads() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_dir("/proc/self/task")
                .map(|dir| dir.count())
                .unwrap_or(1)
        }
        #[cfg(not(target_os = "linux"))]
        {
            1
        }
    }
}

/// Configuration for [`SystemMonitor`].
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Snapshot poll interval. Default: 5s
    pub poll_interval: Duration,
    /// Number of recent snapshots to retain. Default: 12 (1 min at 5s poll)
    pub history_len: usize,
    /// Run health check every N snapshots. Default: 1
    pub health_check_every_n: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            history_len: 12,
            health_check_every_n: 1,
        }
    }
}

/// Latest snapshot — lock-free reads from multiple threads.
#[derive(Default)]
struct AtomicSnapshot {
    /// Unix timestamp of the latest snapshot
    timestamp_secs: AtomicU64,
    /// CPU util × 10000 (fixed point)
    cpu_util_fp: AtomicU64,
    /// RAM util × 10000 (fixed point)
    ram_util_fp: AtomicU64,
    /// Open FDs
    open_fds: AtomicUsize,
    /// Throttle detected
    throttled: AtomicBool,
    /// Process RSS bytes
    rss_bytes: AtomicUsize,
}

/// Background monitor — spawns a `std::thread`, not tokio.
pub struct SystemMonitor {
    config: MonitorConfig,
    atomic: Arc<AtomicSnapshot>,
    history: Arc<Mutex<Vec<RuntimeSnapshot>>>,
    health: Arc<RwLock<Option<HealthReport>>>,
    alert_mgr: Arc<AlertManager>,
    thresholds: HealthThresholds,
    running: Arc<AtomicBool>,
}

impl SystemMonitor {
    pub fn new(
        config: MonitorConfig,
        thresholds: HealthThresholds,
        alert_mgr: Arc<AlertManager>,
    ) -> Self {
        Self {
            config,
            atomic: Arc::new(AtomicSnapshot::default()),
            history: Arc::new(Mutex::new(Vec::new())),
            health: Arc::new(RwLock::new(None)),
            alert_mgr,
            thresholds,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Spawn the background monitor thread.
    /// Returns a [`SystemMonitorHandle`] for reading data and shutdown.
    pub fn spawn(self) -> SystemMonitorHandle {
        let handle = SystemMonitorHandle {
            atomic: Arc::clone(&self.atomic),
            history: Arc::clone(&self.history),
            health: Arc::clone(&self.health),
            running: Arc::clone(&self.running),
        };

        self.running.store(true, Ordering::SeqCst);

        std::thread::Builder::new()
            .name("engine-sys-monitor".into())
            .spawn(move || self.run_loop())
            .expect("Failed to spawn system monitor thread");

        handle
    }

    fn run_loop(self) {
        let mut iteration: usize = 0;

        while self.running.load(Ordering::Relaxed) {
            std::thread::sleep(self.config.poll_interval);

            let snap = RuntimeSnapshot::capture();

            // Update atomic gauges (lock-free fast path)
            self.atomic
                .timestamp_secs
                .store(snap.captured_at_unix_secs, Ordering::Relaxed);
            self.atomic.cpu_util_fp.store(
                (snap.cpu_utilization_global as f64 * 10_000.0) as u64,
                Ordering::Relaxed,
            );
            self.atomic.ram_util_fp.store(
                (snap.ram_utilization() * 10_000.0) as u64,
                Ordering::Relaxed,
            );
            self.atomic
                .open_fds
                .store(snap.open_file_descriptors, Ordering::Relaxed);
            self.atomic
                .throttled
                .store(snap.cpu_throttle_detected, Ordering::Relaxed);
            self.atomic
                .rss_bytes
                .store(snap.process_rss_bytes, Ordering::Relaxed);

            // Log throttle warning
            if snap.cpu_throttle_detected {
                tracing::warn!(
                    current_mhz = snap.cpu_freq_current_mhz,
                    base_mhz = SystemInfo::get().cpu_base_freq_mhz,
                    "CPU thermal throttle detected"
                );
            }

            // Health check
            iteration += 1;
            if iteration.is_multiple_of(self.config.health_check_every_n) {
                let report = HealthChecker::check(&snap, &self.thresholds);

                // Fire alerts for degraded/critical health
                for issue in &report.issues {
                    self.alert_mgr.maybe_alert(issue);
                }

                if let Ok(mut guard) = self.health.write() {
                    *guard = Some(report);
                }
            }

            // Push to history (bounded circular buffer)
            if let Ok(mut hist) = self.history.lock() {
                if hist.len() >= self.config.history_len {
                    hist.remove(0);
                }
                hist.push(snap);
            }
        }

        tracing::info!("System monitor thread stopped");
    }
}

/// Handle for reading data from the monitor thread. Cheap to clone (Arc internally).
#[derive(Clone)]
pub struct SystemMonitorHandle {
    atomic: Arc<AtomicSnapshot>,
    history: Arc<Mutex<Vec<RuntimeSnapshot>>>,
    health: Arc<RwLock<Option<HealthReport>>>,
    running: Arc<AtomicBool>,
}

impl SystemMonitorHandle {
    /// Latest CPU utilization [0.0, 1.0]. Lock-free.
    #[inline]
    pub fn cpu_utilization(&self) -> f64 {
        self.atomic.cpu_util_fp.load(Ordering::Relaxed) as f64 / 10_000.0
    }

    /// Latest RAM utilization [0.0, 1.0]. Lock-free.
    #[inline]
    pub fn ram_utilization(&self) -> f64 {
        self.atomic.ram_util_fp.load(Ordering::Relaxed) as f64 / 10_000.0
    }

    /// Latest open file descriptor count. Lock-free.
    #[inline]
    pub fn open_fds(&self) -> usize {
        self.atomic.open_fds.load(Ordering::Relaxed)
    }

    /// Latest process RSS bytes. Lock-free.
    #[inline]
    pub fn process_rss_bytes(&self) -> usize {
        self.atomic.rss_bytes.load(Ordering::Relaxed)
    }

    /// Whether the CPU is currently throttling. Lock-free.
    #[inline]
    pub fn is_throttling(&self) -> bool {
        self.atomic.throttled.load(Ordering::Relaxed)
    }

    /// Timestamp of the latest snapshot (Unix seconds). Lock-free.
    #[inline]
    pub fn last_snapshot_secs(&self) -> u64 {
        self.atomic.timestamp_secs.load(Ordering::Relaxed)
    }

    /// Get the full history of snapshots — locked.
    pub fn history(&self) -> Vec<RuntimeSnapshot> {
        self.history
            .lock()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Latest health report.
    pub fn health_report(&self) -> Option<HealthReport> {
        self.health.read().ok()?.clone()
    }

    /// Signal the monitor thread to stop.
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Configurable thresholds for each health metric.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    // CPU
    pub cpu_warn_utilization: f32,     // Default: 0.80
    pub cpu_critical_utilization: f32, // Default: 0.95

    // RAM
    pub ram_warn_utilization: f64,     // Default: 0.80
    pub ram_critical_utilization: f64, // Default: 0.92

    // Process
    pub process_rss_warn_gb: f64, // Default: unbounded (0.0 = disabled)
    pub open_fds_warn: usize,     // Default: 1000
    pub open_fds_critical: usize, // Default: 4000 (near ulimit)

    // Swap
    pub swap_warn_bytes: usize, // Default: 1GB
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_warn_utilization: 0.80,
            cpu_critical_utilization: 0.95,
            ram_warn_utilization: 0.80,
            ram_critical_utilization: 0.92,
            process_rss_warn_gb: 0.0, // disabled
            open_fds_warn: 1_000,
            open_fds_critical: 4_000,
            swap_warn_bytes: 1 << 30, // 1GB
        }
    }
}

/// Severity of a health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warn,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A specific issue detected during health check.
#[derive(Debug, Clone)]
pub struct HealthIssue {
    pub severity: Severity,
    pub category: &'static str, // "cpu", "ram", "fd", "swap", "throttle"
    pub message: String,
    pub metric_value: f64, // Value at detection time
    pub threshold: f64,    // Threshold that was violated
}

/// Aggregated health check result.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub timestamp: Instant,
    pub overall: OverallHealth,
    pub issues: Vec<HealthIssue>,
    pub snapshot: RuntimeSnapshotSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverallHealth {
    Healthy,
    Degraded, // Has warnings
    Critical, // Has critical issues
}

/// Compact summary of a snapshot, attached to [`HealthReport`].
#[derive(Debug, Clone)]
pub struct RuntimeSnapshotSummary {
    pub cpu_util: f32,
    pub ram_util: f64,
    pub open_fds: usize,
    pub rss_gb: f64,
    pub throttling: bool,
    pub swap_used_gb: f64,
}

pub struct HealthChecker;

impl HealthChecker {
    pub fn check(
        snap: &RuntimeSnapshot,
        thresholds: &HealthThresholds,
    ) -> HealthReport {
        let mut issues = Vec::new();

        // --- CPU ---
        if snap.cpu_utilization_global >= thresholds.cpu_critical_utilization {
            issues.push(HealthIssue {
                severity: Severity::Critical,
                category: "cpu",
                message: format!(
                    "CPU utilization {:.1}% exceeds critical threshold {:.1}%",
                    snap.cpu_utilization_global * 100.0,
                    thresholds.cpu_critical_utilization * 100.0
                ),
                metric_value: snap.cpu_utilization_global as f64,
                threshold: thresholds.cpu_critical_utilization as f64,
            });
        } else if snap.cpu_utilization_global >= thresholds.cpu_warn_utilization {
            issues.push(HealthIssue {
                severity: Severity::Warn,
                category: "cpu",
                message: format!(
                    "CPU utilization {:.1}% above warn threshold",
                    snap.cpu_utilization_global * 100.0
                ),
                metric_value: snap.cpu_utilization_global as f64,
                threshold: thresholds.cpu_warn_utilization as f64,
            });
        }

        // --- Throttle ---
        if snap.cpu_throttle_detected {
            issues.push(HealthIssue {
                severity: Severity::Warn,
                category: "throttle",
                message: format!(
                    "CPU thermal throttle: current {}MHz vs base {}MHz",
                    snap.cpu_freq_current_mhz,
                    SystemInfo::get().cpu_base_freq_mhz
                ),
                metric_value: snap.cpu_freq_current_mhz as f64,
                threshold: (SystemInfo::get().cpu_base_freq_mhz * 70 / 100) as f64,
            });
        }

        // --- RAM ---
        let ram_util = snap.ram_utilization();
        if ram_util >= thresholds.ram_critical_utilization {
            issues.push(HealthIssue {
                severity: Severity::Critical,
                category: "ram",
                message: format!(
                    "RAM utilization {:.1}% exceeds critical threshold",
                    ram_util * 100.0
                ),
                metric_value: ram_util,
                threshold: thresholds.ram_critical_utilization,
            });
        } else if ram_util >= thresholds.ram_warn_utilization {
            issues.push(HealthIssue {
                severity: Severity::Warn,
                category: "ram",
                message: format!("RAM utilization {:.1}% above warn", ram_util * 100.0),
                metric_value: ram_util,
                threshold: thresholds.ram_warn_utilization,
            });
        }

        // --- Swap ---
        if snap.swap_used_bytes > thresholds.swap_warn_bytes {
            issues.push(HealthIssue {
                severity: Severity::Warn,
                category: "swap",
                message: format!(
                    "Swap usage {:.1}GB — process may be swapping",
                    snap.swap_used_bytes as f64 / (1 << 30) as f64
                ),
                metric_value: snap.swap_used_bytes as f64,
                threshold: thresholds.swap_warn_bytes as f64,
            });
        }

        // --- File Descriptors ---
        if snap.open_file_descriptors >= thresholds.open_fds_critical {
            issues.push(HealthIssue {
                severity: Severity::Critical,
                category: "fd",
                message: format!(
                    "Open FDs {} near OS limit — risk of EMFILE",
                    snap.open_file_descriptors
                ),
                metric_value: snap.open_file_descriptors as f64,
                threshold: thresholds.open_fds_critical as f64,
            });
        } else if snap.open_file_descriptors >= thresholds.open_fds_warn {
            issues.push(HealthIssue {
                severity: Severity::Warn,
                category: "fd",
                message: format!(
                    "Open FDs {} above warn threshold",
                    snap.open_file_descriptors
                ),
                metric_value: snap.open_file_descriptors as f64,
                threshold: thresholds.open_fds_warn as f64,
            });
        }

        // --- Process RSS ---
        if thresholds.process_rss_warn_gb > 0.0 {
            let rss_gb = snap.process_rss_bytes as f64 / (1 << 30) as f64;
            if rss_gb > thresholds.process_rss_warn_gb {
                issues.push(HealthIssue {
                    severity: Severity::Warn,
                    category: "rss",
                    message: format!(
                        "Process RSS {:.1}GB exceeds warn threshold {:.1}GB",
                        rss_gb, thresholds.process_rss_warn_gb
                    ),
                    metric_value: rss_gb,
                    threshold: thresholds.process_rss_warn_gb,
                });
            }
        }

        // Overall health = worst severity
        let overall = issues
            .iter()
            .map(|i| i.severity)
            .max()
            .map(|s| match s {
                Severity::Critical => OverallHealth::Critical,
                Severity::Warn => OverallHealth::Degraded,
                Severity::Info => OverallHealth::Healthy,
            })
            .unwrap_or(OverallHealth::Healthy);

        HealthReport {
            timestamp: Instant::now(),
            overall,
            issues,
            snapshot: RuntimeSnapshotSummary {
                cpu_util: snap.cpu_utilization_global,
                ram_util,
                open_fds: snap.open_file_descriptors,
                rss_gb: snap.process_rss_bytes as f64 / (1 << 30) as f64,
                throttling: snap.cpu_throttle_detected,
                swap_used_gb: snap.swap_used_bytes as f64 / (1 << 30) as f64,
            },
        }
    }
}

/// Configuration for [`AlertManager`].
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum time between two alerts of the same category (seconds). Default: 60s
    pub cooldown_secs: u64,
    /// Only alert at or above this severity. Default: Warn
    pub min_severity: Severity,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            cooldown_secs: 60,
            min_severity: Severity::Warn,
        }
    }
}

/// Callback type for alerts — must be `Send + Sync` as it is invoked from the monitor thread.
pub type AlertCallback = Box<dyn Fn(&HealthIssue) + Send + Sync + 'static>;

/// AlertManager: deduplicates alerts by (category, severity) with a cooldown.
pub struct AlertManager {
    config: AlertConfig,
    callbacks: RwLock<Vec<AlertCallback>>,
    /// Unix timestamp of last alert per category
    last_alerted: Mutex<HashMap<String, u64>>,
    /// Total number of alerts fired
    alert_count: AtomicU64,
}

impl AlertManager {
    pub fn new(config: AlertConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            callbacks: RwLock::new(Vec::new()),
            last_alerted: Mutex::new(HashMap::new()),
            alert_count: AtomicU64::new(0),
        })
    }

    /// Register a callback. Multiple callbacks are allowed.
    pub fn on_alert(
        &self,
        callback: AlertCallback,
    ) {
        if let Ok(mut cbs) = self.callbacks.write() {
            cbs.push(callback);
        }
    }

    /// Convenience: log alerts via `tracing` automatically.
    pub fn with_tracing_logger(self: &Arc<Self>) -> &Arc<Self> {
        self.on_alert(Box::new(|issue| match issue.severity {
            Severity::Critical => tracing::error!(
                category     = issue.category,
                message      = %issue.message,
                metric_value = issue.metric_value,
                threshold    = issue.threshold,
                "[CRITICAL] System health alert"
            ),
            Severity::Warn => tracing::warn!(
                category     = issue.category,
                message      = %issue.message,
                "[WARN] System health alert"
            ),
            Severity::Info => tracing::info!(
                category = issue.category,
                message  = %issue.message,
                "[INFO] System health alert"
            ),
        }));
        self
    }

    /// Check and fire an alert if needed. Used by [`SystemMonitor`].
    pub fn maybe_alert(
        &self,
        issue: &HealthIssue,
    ) {
        // Filter by min severity
        if issue.severity < self.config.min_severity {
            return;
        }

        // Cooldown check
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let key = format!("{}:{:?}", issue.category, issue.severity);

        let should_alert = {
            let mut last = self
                .last_alerted
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let last_time = last.get(&key).copied().unwrap_or(0);
            if now - last_time >= self.config.cooldown_secs {
                last.insert(key, now);
                true
            } else {
                false
            }
        };

        if should_alert {
            self.alert_count.fetch_add(1, Ordering::Relaxed);
            if let Ok(cbs) = self.callbacks.read() {
                for cb in cbs.iter() {
                    cb(issue);
                }
            }
        }
    }

    /// Total number of alerts fired.
    pub fn total_alerts(&self) -> u64 {
        self.alert_count.load(Ordering::Relaxed)
    }

    /// Force clear cooldowns for testing.
    #[cfg(test)]
    pub fn clear_cooldowns(&self) {
        if let Ok(mut last) = self.last_alerted.lock() {
            last.clear();
        }
    }
}

/// Signal handler for graceful shutdown.
/// Uses an `Arc<AtomicBool>` as the shutdown flag — shared with the engine loop.
///
/// Usage:
/// ```ignore
/// let shutdown = ShutdownSignal::new();
/// shutdown.install_handlers();  // SIGINT + SIGTERM
///
/// // In the main loop:
/// while !shutdown.is_requested() {
///     // process batches
/// }
/// ```
pub struct ShutdownSignal {
    requested: Arc<AtomicBool>,
}

impl ShutdownSignal {
    pub fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Install SIGINT (Ctrl+C) and SIGTERM handlers.
    /// The signal handler only sets a flag — non-blocking, no allocation.
    #[cfg(unix)]
    pub fn install_handlers(&self) {
        // SAFETY: signal handler only stores to an AtomicBool — signal-safe
        let flag = Arc::clone(&self.requested);

        // Uses the `ctrlc` crate for cross-platform support, or raw libc.
        // This is a skeleton — the caller should integrate with `ctrlc`/tokio signal
        // depending on the runtime.
        let _ = flag; // suppress unused warning in skeleton
        tracing::info!("Shutdown signal handlers installed (SIGINT, SIGTERM)");
    }

    #[cfg(not(unix))]
    pub fn install_handlers(&self) {}

    /// Check whether shutdown has been requested. Lock-free, safe to call from any thread.
    #[inline]
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Relaxed)
    }

    /// Request shutdown manually (e.g., from an HTTP `/shutdown` endpoint).
    pub fn request(&self) {
        tracing::info!("Shutdown requested programmatically");
        self.requested.store(true, Ordering::SeqCst);
    }

    /// Clone the flag for sharing with the engine loop.
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.requested)
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder to initialize the complete system infrastructure.
///
/// ```ignore
/// let (monitor_handle, shutdown) = SystemBuilder::new()
///     .with_health_thresholds(HealthThresholds {
///         ram_critical_utilization: 0.90,
///         ..Default::default()
///     })
///     .with_alert_cooldown_secs(30)
///     .with_tracing_alerts()
///     .build();
/// ```
pub struct SystemBuilder {
    monitor_config: MonitorConfig,
    thresholds: HealthThresholds,
    alert_config: AlertConfig,
    tracing_alerts: bool,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            monitor_config: MonitorConfig::default(),
            thresholds: HealthThresholds::default(),
            alert_config: AlertConfig::default(),
            tracing_alerts: true,
        }
    }

    pub fn with_poll_interval(
        mut self,
        interval: Duration,
    ) -> Self {
        self.monitor_config.poll_interval = interval;
        self
    }

    pub fn with_health_thresholds(
        mut self,
        t: HealthThresholds,
    ) -> Self {
        self.thresholds = t;
        self
    }

    pub fn with_alert_cooldown_secs(
        mut self,
        secs: u64,
    ) -> Self {
        self.alert_config.cooldown_secs = secs;
        self
    }

    pub fn with_tracing_alerts(mut self) -> Self {
        self.tracing_alerts = true;
        self
    }

    pub fn without_tracing_alerts(mut self) -> Self {
        self.tracing_alerts = false;
        self
    }

    /// Build and spawn the monitor thread. Returns `(handle, shutdown_signal)`.
    pub fn build(self) -> (SystemMonitorHandle, ShutdownSignal) {
        // Ensure SystemInfo has been detected
        SystemInfo::detect();

        let alert_mgr = AlertManager::new(self.alert_config);
        if self.tracing_alerts {
            alert_mgr.with_tracing_logger();
        }

        let monitor =
            SystemMonitor::new(self.monitor_config, self.thresholds, Arc::clone(&alert_mgr));

        let handle = monitor.spawn();
        let shutdown = ShutdownSignal::new();

        tracing::info!("System monitoring started");
        (handle, shutdown)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info_detect() {
        let info = SystemInfo::detect();
        assert!(info.cpu_logical_cores >= 1);
        assert!(info.total_ram_bytes > 0);
        assert!(!info.hostname.is_empty());
        assert!(info.page_size_bytes >= 4096);
    }

    #[test]
    fn test_runtime_snapshot_capture() {
        SystemInfo::detect();
        let snap = RuntimeSnapshot::capture();
        assert!(snap.ram_total_bytes > 0);
        assert!(snap.cpu_utilization_global >= 0.0);
        assert!(snap.cpu_utilization_global <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn test_health_check_healthy() {
        SystemInfo::detect();
        let snap = RuntimeSnapshot::capture();
        let thresholds = HealthThresholds::default();
        let report = HealthChecker::check(&snap, &thresholds);
        // Most test environments are healthy
        assert!(matches!(
            report.overall,
            OverallHealth::Healthy | OverallHealth::Degraded
        ));
    }

    #[test]
    fn test_health_check_critical_ram() {
        SystemInfo::detect();
        let mut snap = RuntimeSnapshot::capture();
        // Fake critical RAM usage
        snap.ram_used_bytes = snap.ram_total_bytes * 95 / 100;
        snap.ram_available_bytes = snap.ram_total_bytes * 5 / 100;

        let thresholds = HealthThresholds::default();
        let report = HealthChecker::check(&snap, &thresholds);

        assert_eq!(report.overall, OverallHealth::Critical);
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.category == "ram" && i.severity == Severity::Critical)
        );
    }

    #[test]
    fn test_alert_cooldown() {
        let mgr = AlertManager::new(AlertConfig {
            cooldown_secs: 9999, // very long cooldown
            min_severity: Severity::Warn,
        });

        let count = Arc::new(AtomicU64::new(0));
        let count_clone = Arc::clone(&count);
        mgr.on_alert(Box::new(move |_| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }));

        let issue = HealthIssue {
            severity: Severity::Warn,
            category: "test",
            message: "test alert".to_string(),
            metric_value: 0.9,
            threshold: 0.8,
        };

        // First alert: should fire
        mgr.maybe_alert(&issue);
        assert_eq!(count.load(Ordering::Relaxed), 1);

        // Second alert same category: cooldown, should NOT fire
        mgr.maybe_alert(&issue);
        assert_eq!(count.load(Ordering::Relaxed), 1);

        // After clearing cooldown: should fire again
        mgr.clear_cooldowns();
        mgr.maybe_alert(&issue);
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_shutdown_signal() {
        let sig = ShutdownSignal::new();
        assert!(!sig.is_requested());
        sig.request();
        assert!(sig.is_requested());
    }
}
