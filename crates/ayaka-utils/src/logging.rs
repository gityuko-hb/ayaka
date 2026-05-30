use std::sync::OnceLock;
use tracing_subscriber::{
    Layer,
    filter::{EnvFilter, LevelFilter},
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

use crate::env::{EnvConfig, LogLevel};

static LOGGING_INITIALIZED: OnceLock<()> = OnceLock::new();

/// RAII guard: flush buffered log entries when dropped (process exit / test teardown).
pub struct LogGuard {
    // non-copy để enforce drop
    _private: (),
}

impl Drop for LogGuard {
    fn drop(&mut self) {
        // Flush all async log sinks before exiting
        // Tracing-subscriber fmt layer automatically flushes when a subscriber is dropped,
        // But if using an OTLP exporter, you need to explicitly flush here.
        opentelemetry::global::shutdown_tracer_provider();
    }
}

// ---- Public API ------------------------------------------------------------

/// Initialize logging. Idempotent — safe to call multiple times.
// Must be called BEFORE spawning any threads, immediately after EnvConfig::init().
pub fn init_logging() -> LogGuard {
    LOGGING_INITIALIZED.get_or_init(|| {
        let cfg = EnvConfig::get();
        build_subscriber(cfg);
        cfg.dump_to_log();
    });
    LogGuard { _private: () }
}

/// Log only once per (file, line) — avoids log flooding.
/// Use for "unchanging info" such as "DType selected is BF16".
#[macro_export]
macro_rules! log_once_info {
    ($($arg:tt)*) => {{
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!($($arg)*);
        }
    }};
}

#[macro_export]
macro_rules! log_once_warn {
    ($($arg:tt)*) => {{
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            tracing::warn!($($arg)*);
        }
    }};
}

/// Create a logger with a fixed prefix for the subsystem.
/// Use: let log = SubsystemLogger::new("scheduler");
///      log.debug("processing batch");
pub struct SubsystemLogger {
    subsystem: &'static str,
}

impl SubsystemLogger {
    pub const fn new(subsystem: &'static str) -> Self {
        Self { subsystem }
    }

    #[inline]
    pub fn trace(
        &self,
        msg: &str,
    ) {
        tracing::trace!(subsystem = self.subsystem, "{}", msg);
    }

    #[inline]
    pub fn debug(
        &self,
        msg: &str,
    ) {
        tracing::debug!(subsystem = self.subsystem, "{}", msg);
    }

    #[inline]
    pub fn info(
        &self,
        msg: &str,
    ) {
        tracing::info!(subsystem = self.subsystem, "{}", msg);
    }

    #[inline]
    pub fn warn(
        &self,
        msg: &str,
    ) {
        tracing::warn!(subsystem = self.subsystem, "{}", msg);
    }

    #[inline]
    pub fn error(
        &self,
        msg: &str,
    ) {
        tracing::error!(subsystem = self.subsystem, "{}", msg);
    }
}

/// Create a span to measure the duration of an operation.
/// Automatically log the duration when dropped.
///
/// ```rust
/// use ayaka_utils::timed_span;
/// let _span = timed_span!("prefill", "tokens" = "512");
/// // ... do prefill ...
/// // when _span drop: logs "prefill completed in Xms"
/// ```
#[macro_export]
macro_rules! timed_span {
    ($name:expr) => {
        tracing::info_span!($name).entered()
    };
    ($name:expr, $($field:tt)*) => {
        tracing::info_span!($name, $($field)*).entered()
    };
}

fn build_subscriber(cfg: &EnvConfig) {
    // Build EnvFilter từ ENGINE_LOG_LEVEL + per-subsystem overrides
    let mut filter = EnvFilter::builder()
        .with_default_directive(to_level_filter(cfg.log_level).into())
        .from_env_lossy();

    // Noise suppression — third-party crates or flood log
    for directive in SUPPRESS_DIRECTIVES {
        filter = filter.add_directive(directive.parse().unwrap());
    }

    // Per-subsystem overrides from ENGINE_LOG_<SUBSYSTEM>=<level>
    for (subsystem, level) in &cfg.log_overrides {
        let directive = format!("engine_{}={}", subsystem, level.as_str());
        if let Ok(d) = directive.parse() {
            filter = filter.add_directive(d);
        }
    }

    if cfg.log_json {
        // JSON mode: for production, k8s log aggregation (Loki, CloudWatch, ...)
        let layer = fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(false)
            .with_target(true)
            .with_thread_ids(false)
            .with_span_events(if cfg.debug {
                FmtSpan::ENTER | FmtSpan::CLOSE
            } else {
                FmtSpan::CLOSE
            });

        tracing_subscriber::registry()
            .with(layer.with_filter(filter))
            .init();
    } else {
        // Text mode: for development
        let layer = fmt::layer()
            .with_target(cfg.debug)
            .with_thread_ids(cfg.debug)
            .with_file(cfg.debug)
            .with_line_number(cfg.debug)
            .with_span_events(if cfg.debug {
                FmtSpan::ENTER | FmtSpan::CLOSE
            } else {
                FmtSpan::NONE
            });

        tracing_subscriber::registry()
            .with(layer.with_filter(filter))
            .init();
    }
}

fn to_level_filter(level: LogLevel) -> LevelFilter {
    match level {
        LogLevel::Trace => LevelFilter::TRACE,
        LogLevel::Debug => LevelFilter::DEBUG,
        LogLevel::Info => LevelFilter::INFO,
        LogLevel::Warn => LevelFilter::WARN,
        LogLevel::Error => LevelFilter::ERROR,
    }
}

/// Third-party crates need to be suppressed — concentrate them here, don't scatter them.
const SUPPRESS_DIRECTIVES: &[&str] = &[
    "symphonia=warn",
    "tokenizers=warn",
    "reqwest=warn",
    "hyper=warn",
    "h2=warn",
    "rustls=warn",
    "hf_hub=warn",
    "candle_core=warn",
];

mod opentelemetry {
    pub mod global {
        pub fn shutdown_tracer_provider() {
            // No-op when the otlp feature is not enabled.
            // When the feature is enabled: replace with opentelemetry::global::shutdown_tracer_provider()
        }
    }
}
