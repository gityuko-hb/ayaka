#![allow(unexpected_cfgs)]

//! Reference: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/utils/memory_usage.rs
use ayaka_core::error::EngineError;
use candle_core::Device;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use sysinfo::System;

#[cfg(feature = "metal")]
use tracing::warn;
#[cfg(feature = "metal")]
use crate::constants::BYTES_PER_MIB;

#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    pub total_bytes: usize,
    pub available_bytes: usize,
    pub is_unified: bool,
}

impl MemorySnapshot {
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.total_bytes
            .saturating_sub(self.available_bytes)
    }

    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes() as f64 / self.total_bytes as f64
    }

    #[inline]
    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / (1 << 30) as f64
    }

    #[inline]
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1 << 30) as f64
    }
}

/// Memory type — discrete (separate VRAM) or unified (Apple Silicon, NVIDIA Grace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// GPU has its own VRAM — CPU and GPU do not share physical memory.
    Discrete,

    // Unified memory — CPU and GPU share the same physical pool.
    // Loading tensors onto the CPU before moving them to the GPU doesn't save anything.
    Unified,
}

/// Low-level query helper — not used directly by the caller,
/// but via MemoryPlanner or MultiDeviceMemory.
pub(crate) struct MemoryUsage;

impl MemoryUsage {
    pub fn query(
        &self,
        device: &Device,
    ) -> Result<MemorySnapshot, EngineError> {
        match device {
            Device::Cpu => self.query_cpu(),

            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if Self::is_integrated_cuda(device) {
                    self.query_integrated_cuda()
                } else {
                    self.query_discrete_cuda(dev)
                }
            },

            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(EngineError::UnsupportedDevice("cuda".into())),

            #[cfg(feature = "metal")]
            Device::Metal(dev) => self.query_metal(dev),

            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => Err(EngineError::UnsupportedDevice("metal".into())),
        }
    }

    fn query_cpu(&self) -> Result<MemorySnapshot, EngineError> {
        let sys = System::new_all();
        Ok(MemorySnapshot {
            total_bytes: sys.total_memory() as usize,
            available_bytes: sys.available_memory() as usize,
            is_unified: false,
        })
    }

    #[cfg(feature = "cuda")]
    fn query_discrete_cuda(
        &self,
        dev: &candle_core::CudaDevice,
    ) -> Result<MemorySnapshot, EngineError> {
        use candle_core::cuda::cudarc::driver::result;
        use candle_core::cuda_backend::WrapErr;

        dev.cuda_stream()
            .context()
            .bind_to_thread()
            .w()
            .map_err(|e| EngineError::Internal {
                msg: e.to_string(),
                location: concat!(file!(), ":", line!()).to_string(),
            })?;

        let (free, total) =
            result::mem_get_info()
                .w()
                .map_err(|e| EngineError::Internal {
                    msg: e.to_string(),
                    location: concat!(file!(), ":", line!()).to_string(),
                })?;

        Ok(MemorySnapshot {
            total_bytes: total,
            available_bytes: free,
            is_unified: false,
        })
    }

    #[cfg(feature = "cuda")]
    fn query_integrated_cuda(&self) -> Result<MemorySnapshot, EngineError> {
        use crate::env::EnvConfig;
        let sys = System::new_all();
        let total = sys.total_memory() as usize;
        let avail = sys.available_memory() as usize;
        let fraction = EnvConfig::get().igpu_memory_fraction;
        let budget = (total as f64 * fraction) as usize;
        let used_est = total.saturating_sub(avail);
        let avail_est = budget.saturating_sub((used_est as f64 * fraction) as usize);
        Ok(MemorySnapshot {
            total_bytes: budget,
            available_bytes: avail_est,
            is_unified: true,
        })
    }

    #[cfg(feature = "cuda")]
    fn is_integrated_cuda(device: &Device) -> bool {
        use candle_core::cuda::cudarc::driver::{result, sys};
        if let Device::Cuda(dev) = device {
            let ordinal = dev.cuda_stream().context().ordinal();
            if let Ok(cu_dev) = result::device::get(ordinal as i32) {
                return unsafe {
                    result::device::get_attribute(
                        cu_dev,
                        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED,
                    )
                    .map(|v| v != 0)
                    .unwrap_or(false)
                };
            }
        }
        false
    }

    #[cfg(feature = "metal")]
    fn query_metal(
        &self,
        dev: &candle_core::MetalDevice,
    ) -> Result<MemorySnapshot, EngineError> {
        let sysctl_floor = self.metal_sysctl_floor()?;
        let device_max = dev.device().recommended_max_working_set_size() as usize;
        let budget = sysctl_floor.max(device_max);

        // Metal warning: recommendedMaxWorkingSetSize underreports pada when
        // system memory pressure tinggi. Threshold configurable (default /2).
        if device_max < sysctl_floor / 2 {
            warn!(
                device_max_mb = device_max / BYTES_PER_MIB,
                sysctl_floor_mb = sysctl_floor / BYTES_PER_MIB,
                "Metal recommendedMaxWorkingSetSize severely underreporting — using sysctl floor"
            );
        }

        Ok(MemorySnapshot {
            total_bytes: budget,
            available_bytes: budget.saturating_sub(allocated),
            is_unified: true,
        })
    }

    #[cfg(feature = "metal")]
    fn metal_sysctl_floor(&self) -> Result<usize, EngineError> {
        let sys = System::new_all();
        let sys_mb = sys.total_memory() / BYTES_PER_MIB;

        // Read iogpu.wired_limit_mb from sysctl
        let sysctl_mb = std::process::Command::new("sysctl")
            .args(["-n", "iogpu.wired_limit_mb"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<usize>().ok());

        // Default cap according to RAM size (same logic as mistral.rs)
        let default_cap_mb = match sys_mb {
            x if x <= 36 * 1024 => (sys_mb * 2) / 3,
            _ => (sys_mb * 3) / 4,
        };

        let floor_mb = match sysctl_mb {
            Some(0) | None => default_cap_mb,
            Some(x) => x as u64,
        };

        Ok(floor_mb * BYTES_PER_MIB)
    }
}

/// MemoryPlan - Memory allocation plan after loading model weights.
/// This is the input to calculate the number of KV cache blocks for PagedAttention.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub device_label: String,
    pub kind: MemoryKind,
    pub total_bytes: usize,
    /// Bytes were used for model weights (measured after loading).
    pub model_weights_bytes: usize,
    /// Estimate activation buffer (batch_tokens × hidden × layers × 2)
    pub activation_buffer_bytes: usize,
    /// Headroom to avoid CUDA OOM due to fragmentation (5% total)
    pub reserved_bytes: usize,
    /// The rest: reserved for KV cache
    pub kv_cache_bytes: usize,
}

impl MemoryPlan {
    /// Maximum number of KV blocks that can be allocated (PagedAttention).
    ///
    /// `block_bytes` = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes × block_size
    #[inline]
    pub fn max_kv_blocks(
        &self,
        block_bytes: usize,
    ) -> usize {
        if block_bytes == 0 {
            return 0;
        }
        self.kv_cache_bytes / block_bytes
    }

    /// Log the entire plan — call it after the calculation is complete.
    pub fn log_summary(&self) {
        let gb = |b: usize| b as f64 / (1 << 30) as f64;
        tracing::info!(
            device          = %self.device_label,
            kind            = ?self.kind,
            total_gb        = format_args!("{:.2}", gb(self.total_bytes)),
            weights_gb      = format_args!("{:.2}", gb(self.model_weights_bytes)),
            activation_gb   = format_args!("{:.2}", gb(self.activation_buffer_bytes)),
            reserved_gb     = format_args!("{:.2}", gb(self.reserved_bytes)),
            kv_cache_gb     = format_args!("{:.2}", gb(self.kv_cache_bytes)),
            "Memory plan finalized"
        );
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPlannerConfig {
    /// Fraction total memory for the engine [0.1, 0.98]
    pub gpu_memory_fraction: f64,
    /// Fraction for unified memory (iGPU)
    pub igpu_memory_fraction: f64,
    /// Reserved headroom fraction (default 0.05 = 5%)
    pub reserved_fraction: f64,
}

impl Default for MemoryPlannerConfig {
    fn default() -> Self {
        use crate::env::EnvConfig;
        let cfg = EnvConfig::get();
        Self {
            gpu_memory_fraction: cfg.gpu_memory_fraction,
            igpu_memory_fraction: cfg.igpu_memory_fraction,
            reserved_fraction: 0.05,
        }
    }
}

pub struct MemoryPlanner {
    config: MemoryPlannerConfig,
}

impl MemoryPlanner {
    pub fn new(config: MemoryPlannerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(MemoryPlannerConfig::default())
    }

    /// Estimate the memory needed to load the model BEFORE actually loading.
    /// Used to fail fast if there isn't enough VRAM.
    /// 
    /// `param_count`: Total number of model parameters
    /// `dtype_bytes`: Bytes per element (examples: 2 for BF16/F16, 4 for F32)
    pub fn estimate_model_bytes(
        param_count: u64,
        dtype_bytes: usize,
    ) -> usize {
        // Overhead: 10% extra bytes for optimizer states, buffers, fragmentation
        let base = param_count as usize * dtype_bytes;
        base + base / 10
    }

    /// Check before loading: Does the device have enough memory?
    /// Return a clear error instead of an OOM message in the middle of the process.
    pub fn preflight_check(
        &self,
        device: &Device,
        estimated_model_bytes: usize,
    ) -> Result<(), EngineError> {
        let snapshot = MemoryUsage.query(device)?;
        let budget = self.budget_bytes(&snapshot);

        if estimated_model_bytes > budget {
            return Err(EngineError::OutOfMemoryDevice {
                device: format!("{:?}", device.location()),
                needed_mb: estimated_model_bytes / (1 << 20),
                available_mb: budget / (1 << 20),
            });
        }

        tracing::info!(
            estimated_gb = format_args!("{:.2}", estimated_model_bytes as f64 / (1 << 30) as f64),
            available_gb = format_args!("{:.2}", snapshot.available_gb()),
            "Preflight memory check passed"
        );
        Ok(())
    }

    #[inline]
    fn budget_bytes(
        &self,
        snapshot: &MemorySnapshot,
    ) -> usize {
        let fraction = if snapshot.is_unified {
            self.config.igpu_memory_fraction
        } else {
            self.config.gpu_memory_fraction
        };
        (snapshot.total_bytes as f64 * fraction) as usize
    }

    pub fn plan(
        &self,
        device: &Device,
        measured_weights_bytes: usize,
        max_batch_tokens: usize, // max_batch_size × max_seq_len
        hidden_size: usize,
        num_layers: usize,
        dtype_bytes: usize,
    ) -> Result<MemoryPlan, EngineError> {
        let snapshot = MemoryUsage.query(device)?;
        let budget = self.budget_bytes(&snapshot);

        // Activation buffer estimate:
        // Worst case = max_batch_tokens × hidden_size × dtype_bytes × num_layers × 2
        // Factor 2: intermediate activations trong attention + FFN
        let activation_buffer_bytes = max_batch_tokens
            .saturating_mul(hidden_size)
            .saturating_mul(dtype_bytes)
            .saturating_mul(num_layers)
            .saturating_mul(2);
        // Reserved headroom: avoid OOM due to CUDA allocator fragmentation
        let reserved_bytes = (snapshot.total_bytes as f64 * self.config.reserved_fraction) as usize;

        let overhead = measured_weights_bytes
            .saturating_add(activation_buffer_bytes)
            .saturating_add(reserved_bytes);

        let kv_cache_bytes = budget.saturating_sub(overhead);
        if kv_cache_bytes == 0 {
            tracing::warn!(
                "KV cache budget is 0 — model weights consume entire budget. \
                 Consider reducing gpu_memory_fraction or using a smaller model."
            );
        }

        let plan = MemoryPlan {
            device_label: format!("{:?}", device.location()),
            kind: if snapshot.is_unified {
                MemoryKind::Unified
            } else {
                MemoryKind::Discrete
            },
            total_bytes: snapshot.total_bytes,
            model_weights_bytes: measured_weights_bytes,
            activation_buffer_bytes,
            reserved_bytes,
            kv_cache_bytes,
        };

        plan.log_summary();
        Ok(plan)
    }
}

// MULTI-DEVICE MEMORY
/// Aggregate memory information from multiple devices simultaneously.
/// Used for parallel tensor / parallel pipeline deployment.
#[derive(Debug)]
pub struct MultiDeviceMemory {
    snapshots: Vec<(String, MemorySnapshot)>,
}

impl MultiDeviceMemory {
    pub fn query_all(devices: &[Device]) -> Result<Self, EngineError> {
        let mut snapshots = Vec::with_capacity(devices.len());
        for device in devices {
            let label = format!("{:?}", device.location());
            let snapshot = MemoryUsage.query(device)?;
            snapshots.push((label, snapshot));
        }
        Ok(Self { snapshots })
    }

    /// Total available bytes across all devices.
    pub fn total_available_bytes(&self) -> usize {
        self.snapshots
            .iter()
            .map(|(_, s)| s.available_bytes)
            .sum()
    }

    /// The device with the least memory is the system bottleneck.
    pub fn bottleneck(&self) -> Option<(&str, &MemorySnapshot)> {
        self.snapshots
            .iter()
            .min_by_key(|(_, s)| s.available_bytes)
            .map(|(l, s)| (l.as_str(), s))
    }

    pub fn log_all(&self) {
        for (label, snap) in &self.snapshots {
            tracing::info!(
                device       = %label,
                total_gb     = format_args!("{:.2}", snap.total_gb()),
                available_gb = format_args!("{:.2}", snap.available_gb()),
                utilization  = format_args!("{:.1}%", snap.utilization() * 100.0),
                unified      = snap.is_unified,
                "Device memory"
            );
        }
    }
}

/// Threshold configuration for pressure callbacks.
#[derive(Debug, Clone, Copy)]
pub struct PressureThresholds {
    /// Log warning when utilization exceeds this level. Default: 0.85
    pub warn_utilization: f64,
    /// Call pressure_callback when overtaking. Default: 0.92
    pub critical_utilization: f64,
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            warn_utilization: 0.85,
            critical_utilization: 0.92,
        }
    }
}

/// Shared counter — The scheduler will read the data to determine if KV blocks should be excluded.
#[derive(Debug, Default)]
pub struct MemoryPressureGauge {
    /// 0 = normal, 1 = warn, 2 = critical
    level: AtomicUsize,
}

impl MemoryPressureGauge {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    #[inline]
    pub fn level(&self) -> u8 {
        self.level.load(Ordering::Relaxed) as u8
    }

    #[inline]
    pub fn is_critical(&self) -> bool {
        self.level() >= 2
    }

    #[inline]
    pub fn set_level(
        &self,
        level: u8,
    ) {
        self.level
            .store(level as usize, Ordering::Relaxed);
    }
}

/// Background thread polls memory and updates gauge.
/// Use std::thread because async — pure polling loop is not needed.
pub struct MemoryMonitor {
    gauge: Arc<MemoryPressureGauge>,
    thresholds: PressureThresholds,
    interval: std::time::Duration,
}

impl MemoryMonitor {
    pub fn new(
        gauge: Arc<MemoryPressureGauge>,
        thresholds: PressureThresholds,
        interval: std::time::Duration,
    ) -> Self {
        Self {
            gauge,
            thresholds,
            interval,
        }
    }

    /// Spawn background monitor thread.
    /// Return JoinHandle — caller hold to shut down gracefully.
    pub fn spawn(
        self,
        device: Device,
    ) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("engine-mem-monitor".into())
            .spawn(move || self.run(device))
            .expect("failed to spawn memory monitor thread")
    }

    fn run(
        self,
        device: Device,
    ) {
        loop {
            std::thread::sleep(self.interval);
            match MemoryUsage.query(&device) {
                Ok(snap) => {
                    let util = snap.utilization();
                    let level = if util >= self.thresholds.critical_utilization {
                        tracing::warn!(
                            utilization = format_args!("{:.1}%", util * 100.0),
                            "Memory pressure CRITICAL"
                        );
                        2
                    } else if util >= self.thresholds.warn_utilization {
                        tracing::warn!(
                            utilization = format_args!("{:.1}%", util * 100.0),
                            "Memory pressure HIGH"
                        );
                        1
                    } else {
                        0
                    };
                    self.gauge.set_level(level);
                },
                Err(e) => {
                    tracing::error!(error = %e, "Memory monitor query failed");
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_utilization() {
        let snap = MemorySnapshot {
            total_bytes: 16 * (1 << 30),
            available_bytes: 4 * (1 << 30),
            is_unified: false,
        };
        let util = snap.utilization();
        assert!((util - 0.75).abs() < 1e-6);
        assert!((snap.total_gb() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_plan_kv_blocks() {
        let plan = MemoryPlan {
            device_label: "cuda:0".into(),
            kind: MemoryKind::Discrete,
            total_bytes: 80 * (1 << 30),
            model_weights_bytes: 14 * (1 << 30),
            activation_buffer_bytes: 2 * (1 << 30),
            reserved_bytes: 4 * (1 << 30),
            kv_cache_bytes: 60 * (1 << 30),
        };
        // block_bytes = 32 layers × 2 × 32 heads × 128 dim × 2 bytes × 16 tokens
        let block_bytes = 32 * 2 * 32 * 128 * 2 * 16; // = 8,388,608 = 8MB
        let blocks = plan.max_kv_blocks(block_bytes);
        assert!(blocks > 0);
        // 60GB / 8MB = 7680 blocks
        assert_eq!(blocks, 60 * 1024 * 1024 * 1024 / block_bytes);
    }

    #[test]
    fn test_pressure_gauge() {
        let gauge = MemoryPressureGauge::new();
        assert_eq!(gauge.level(), 0);
        assert!(!gauge.is_critical());

        gauge.set_level(2);
        assert!(gauge.is_critical());
    }
}
