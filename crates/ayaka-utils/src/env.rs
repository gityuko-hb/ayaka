use std::collections::HashMap;
use std::sync::OnceLock;

static GLOBAL_ENV_CONFIG: OnceLock<EnvConfig> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct EnvConfig {
    // --- Logging ---
    /// Default log level. Default: "info"
    /// Override: ENGINE_LOG_LEVEL=debug
    pub log_level: LogLevel,

    /// Per-subsystem log level overrides.
    /// Override: ENGINE_LOG_SCHEDULER=debug, ENGINE_LOG_CUDA=warn
    pub log_overrides: HashMap<String, LogLevel>,

    /// Enable structured JSON logging (for production/k8s).
    /// Override: ENGINE_LOG_JSON=1
    pub log_json: bool,

    /// Turn on debug mode: add file:line, thread IDs, verbose output.
    /// Override: ENGINE_DEBUG=1
    pub debug: bool,

    // --- Memory ---
    /// Fraction total VRAM for engine [0.1, 0.98]. Default: 0.90
    /// Override: ENGINE_GPU_MEMORY_FRACTION=0.85
    pub gpu_memory_fraction: f64,

    /// Fraction RAM for iGPU unified memory [0.1, 0.95]. Default: 0.75
    /// Override: ENGINE_IGPU_MEMORY_FRACTION=0.70
    pub igpu_memory_fraction: f64,

    /// CPU swap space (GB) for offloading. Default: 0
    /// Override: ENGINE_CPU_SWAP_GB=4
    pub cpu_swap_gb: usize,

    // --- Loading ---
    /// Turn off mmap when loading safetensors (used when the filesystem does not support mmap).
    /// Override: ENGINE_NO_MMAP=1
    pub no_mmap: bool,

    /// Number of threads for tensor loading. Default: num_cpus
    /// Override: ENGINE_LOAD_THREADS=8
    pub load_threads: usize,

    // --- Serving ---
    /// Override max batch size from config file.
    /// Override: ENGINE_MAX_BATCH_SIZE=32
    pub max_batch_size: Option<usize>,

    /// Override max sequence length.
    /// Override: ENGINE_MAX_SEQ_LEN=4096
    pub max_seq_len: Option<usize>,

    /// Seed for sampling. Default: None (random)
    /// Override: ENGINE_SEED=42
    pub seed: Option<u64>,

    // --- PagedAttention ---
    /// Number of tokens per KV cache block. Default: 16
    /// Override: ENGINE_KV_BLOCK_SIZE=32
    pub kv_block_size: usize,

    // --- Scheduling ---
    /// Max tokens per batch iteration (prefill worst-case). Default: 2048
    /// Override: ENGINE_MAX_NUM_BATCHED_TOKENS=4096
    pub max_num_batched_tokens: usize,

    // --- Profiling ---
    /// Enable CUDA/Metal profiler markers.
    /// Override: ENGINE_PROFILE=1
    pub profiling_enabled: bool,

    /// Interval dump metrics to log (seconds). 0 = off. Default: 30
    /// Override: ENGINE_METRICS_INTERVAL=60
    pub metrics_interval_secs: u64,
}

impl EnvConfig {
    // Parse and validate all environment variables.
    // Call only once at main() before spawning any threads.
    // Panic immediately if an invalid value is found — fail fast, do not run with incorrect configuration.
    pub fn init() -> &'static Self {
        GLOBAL_ENV_CONFIG.get_or_init(|| {
            let cfg = Self::parse();
            cfg.validate();
            cfg
        })
    }

    // Get the parsed configuration. Panic if init() hasn't been called yet.
    #[inline]
    pub fn get() -> &'static Self {
        GLOBAL_ENV_CONFIG.get().expect(
            "EnvConfig not initialized — call EnvConfig::init() at startup before spawning threads",
        )
    }

    // Use in tests: inject configuration without setting actual environment variables.

    /// WARNING: Use only in #[cfg(test)] — not in production code.
    #[cfg(test)]
    pub fn with_overrides(overrides: impl FnOnce(&mut EnvConfig)) -> EnvConfig {
        let mut cfg = Self::parse();
        overrides(&mut cfg);
        cfg.validate();
        cfg
    }

    // Dump the entire config to tracing — call after initial logging.
    pub fn dump_to_log(&self) {
        tracing::info!(
            log_level   = ?self.log_level,
            log_json    = self.log_json,
            debug       = self.debug,
            gpu_mem_frac = self.gpu_memory_fraction,
            igpu_mem_frac = self.igpu_memory_fraction,
            cpu_swap_gb = self.cpu_swap_gb,
            no_mmap     = self.no_mmap,
            load_threads = self.load_threads,
            max_batch   = ?self.max_batch_size,
            max_seq_len = ?self.max_seq_len,
            seed        = ?self.seed,
            kv_block_sz = self.kv_block_size,
            max_batched_tokens = self.max_num_batched_tokens,
            profiling   = self.profiling_enabled,
            metrics_interval = self.metrics_interval_secs,
            "Engine environment config"
        );
        for (subsystem, level) in &self.log_overrides {
            tracing::info!(subsystem, level = ?level, "Log level override");
        }
    }

    fn parse() -> Self {
        let load_threads = env_usize(crate::constants::ENV_LOAD_THREADS, 0)
            .max(1)  // ít nhất 1
            // 0 = auto-detect from num_cpus.
            // shouldn't exceed the number of logical CPUs too much to avoid thrashing.
            // Supervisors will issue a warning if the CPU count exceeds 4x.
            .max(if env_usize(crate::constants::ENV_LOAD_THREADS, 0) == 0 {
                num_cpus_approx()
            } else {
                0
            });

        use crate::constants::*;

        Self {
            log_level: env_log_level(ENV_LOG_LEVEL, LogLevel::Info),
            log_overrides: parse_log_overrides(),
            log_json: env_bool(ENV_LOG_JSON, false),
            debug: env_bool(ENV_DEBUG, false),
            gpu_memory_fraction: env_f64_range(
                ENV_GPU_MEMORY_FRACTION,
                DEFAULT_GPU_MEMORY_FRACTION,
                GPU_MEMORY_FRACTION_MIN,
                GPU_MEMORY_FRACTION_MAX,
            ),
            igpu_memory_fraction: env_f64_range(
                ENV_IGPU_MEMORY_FRACTION,
                DEFAULT_IGPU_MEMORY_FRACTION,
                IGPU_MEMORY_FRACTION_MIN,
                IGPU_MEMORY_FRACTION_MAX,
            ),
            cpu_swap_gb: env_usize(ENV_CPU_SWAP_GB, 0),
            no_mmap: env_bool(ENV_NO_MMAP, false),
            load_threads,
            max_batch_size: env_opt_usize(ENV_MAX_BATCH_SIZE),
            max_seq_len: env_opt_usize(ENV_MAX_SEQ_LEN),
            seed: env_opt_u64(ENV_SEED),
            kv_block_size: env_usize(ENV_KV_BLOCK_SIZE, DEFAULT_KV_BLOCK_SIZE),
            max_num_batched_tokens: env_usize(
                crate::constants::ENV_MAX_NUM_BATCHED_TOKENS,
                crate::constants::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            ),
            profiling_enabled: env_bool(ENV_PROFILE, false),
            metrics_interval_secs: env_u64(ENV_METRICS_INTERVAL, DEFAULT_METRICS_INTERVAL_SECS),
        }
    }

    fn validate(&self) {
        // kv_block_size must be a power of 2 for correct alignment.
        assert!(
            self.kv_block_size.is_power_of_two()
                && self.kv_block_size >= crate::constants::KV_BLOCK_SIZE_MIN,
            "ENGINE_KV_BLOCK_SIZE={} must be a power of 2 >= {}",
            self.kv_block_size,
            crate::constants::KV_BLOCK_SIZE_MIN,
        );
        // max_num_batched_tokens must be positive
        assert!(
            self.max_num_batched_tokens > 0,
            "ENGINE_MAX_NUM_BATCHED_TOKENS={} must be > 0",
            self.max_num_batched_tokens,
        );
        // Load_threads should not exceed the number of logical CPUs by too much.
        let cpus = num_cpus_approx();
        if self.load_threads > cpus * crate::constants::LOAD_THREADS_THRASHING_RATIO {
            eprintln!(
                "WARNING: ENGINE_LOAD_THREADS={} is much larger than CPU count ({}). \
                 This may cause thrashing.",
                self.load_threads, cpus
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(Self::Trace),
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            other => Err(format!("Unknown log level: {other}")),
        }
    }
}

fn env_bool(
    key: &str,
    default: bool,
) -> bool {
    std::env::var(key)
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_usize(
    key: &str,
    default: usize,
) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(
    key: &str,
    default: u64,
) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_opt_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
}

fn env_opt_u64(key: &str) -> Option<u64> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
}

fn env_f64_range(
    key: &str,
    default: f64,
    min: f64,
    max: f64,
) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|f| {
            if f < min || f > max {
                eprintln!("WARNING: {key}={f} is outside [{min}, {max}], using default {default}");
                default
            } else {
                f
            }
        })
        .unwrap_or(default)
}

fn env_log_level(
    key: &str,
    default: LogLevel,
) -> LogLevel {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<LogLevel>().ok())
        .unwrap_or(default)
}

fn parse_log_overrides() -> HashMap<String, LogLevel> {
    let mut map = HashMap::new();
    for (key, val) in std::env::vars() {
        if let Some(subsystem) = key.strip_prefix(crate::constants::ENV_LOG_PREFIX) {
            if subsystem == "LEVEL" || subsystem == "JSON" {
                continue;
            }
            if let Ok(level) = val.trim().parse::<LogLevel>() {
                map.insert(subsystem.to_lowercase(), level);
            }
        }
    }
    map
}

fn num_cpus_approx() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(crate::constants::DEFAULT_CPU_COUNT_FALLBACK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_overrides() {
        let cfg = EnvConfig::with_overrides(|c| {
            c.gpu_memory_fraction = 0.80;
            c.kv_block_size = 32;
            c.debug = true;
        });
        assert_eq!(cfg.gpu_memory_fraction, 0.80);
        assert_eq!(cfg.kv_block_size, 32);
        assert!(cfg.debug);
    }

    #[test]
    fn test_log_level_parse() {
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("WARN".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("warning".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_kv_block_size_power_of_two() {
        let cfg = EnvConfig::with_overrides(|c| c.kv_block_size = 16);
        assert_eq!(cfg.kv_block_size, 16);
    }

    #[test]
    fn test_max_num_batched_tokens_default() {
        let cfg = EnvConfig::with_overrides(|c| {
            c.max_num_batched_tokens = 4096;
        });
        assert_eq!(cfg.max_num_batched_tokens, 4096);
    }
}
