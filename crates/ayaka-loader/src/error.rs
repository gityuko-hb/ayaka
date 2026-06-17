//! Error type for the loader pipeline.

use std::path::PathBuf;

/// Errors raised while parsing metadata, reading weight files, or planning a load.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("io error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {what}: {source}")]
    Parse {
        what: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("malformed GGUF file: {0}")]
    Gguf(String),

    #[error("unsupported model architecture: {0}")]
    UnsupportedArch(String),

    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    /// GPU memory is exhausted for the attempted strategy. Carries the shortfall
    /// so the orchestrator and logs can report it. Raised proactively by the
    /// loader (estimate vs. driver free) and matched reactively from candle OOM
    /// so `strategy::gpu::load_with_fallback` can degrade
    /// `FullGpu → PartialOffload → SequentialStream` instead of panicking.
    #[error("insufficient VRAM: need {needed} bytes but only {free} bytes free")]
    OutOfVram { needed: usize, free: usize },

    /// The dedicated CUDA-driver thread is unavailable (failed to spawn, or its
    /// channel closed). VRAM queries cannot be served.
    #[error("CUDA driver thread is unavailable")]
    DriverThreadDown,

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("memory error: {0}")]
    Memory(#[from] ayaka_error::AyakaError),

    #[error(transparent)]
    Dequant(#[from] ayaka_quant::GgufDequantError),
}

impl LoaderError {
    /// True when the error signals GPU memory exhaustion — i.e. retrying with a
    /// more conservative strategy (`FullGpu → PartialOffload →
    /// SequentialStream`) might succeed.
    ///
    /// Matches three sources:
    ///   1. our explicit [`LoaderError::OutOfVram`];
    ///   2. `ayaka_error::AyakaError` carrying [`ErrorKind::OutOfMemory`]
    ///      (e.g. a failed `ayaka_memory` device allocation);
    ///   3. candle/cudarc OOM, which only surfaces as a message string.
    ///
    /// The substring match in (3) is a heuristic; it can be tightened to match
    /// cudarc's `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` code directly once that
    /// is threaded through candle.
    pub fn is_vram_oom(&self) -> bool {
        match self {
            LoaderError::OutOfVram { .. } => true,
            LoaderError::Memory(e) => e.kind() == ayaka_error::ErrorKind::OutOfMemory,
            LoaderError::Candle(e) => {
                let msg = e.to_string().to_ascii_lowercase();
                msg.contains("out of memory")
                    || msg.contains("cuda_error_out_of_memory")
                    || msg.contains("oom")
            },
            _ => false,
        }
    }
}

/// Convenient result alias for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_vram_is_classified_as_oom() {
        let e = LoaderError::OutOfVram {
            needed: 1024,
            free: 512,
        };
        assert!(e.is_vram_oom());
    }

    #[test]
    fn candle_oom_message_is_classified() {
        let e = LoaderError::Candle(candle_core::Error::Msg(
            "CUDA_ERROR_OUT_OF_MEMORY: out of memory".to_string(),
        ));
        assert!(e.is_vram_oom());
    }

    #[test]
    fn non_oom_errors_are_not_misclassified() {
        let e = LoaderError::InvalidConfig("missing field".to_string());
        assert!(!e.is_vram_oom());
        let e = LoaderError::Gguf("bad magic".to_string());
        assert!(!e.is_vram_oom());
    }
}
