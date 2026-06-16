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

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("memory error: {0}")]
    Memory(#[from] ayaka_error::AyakaError),

    #[error(transparent)]
    Dequant(#[from] ayaka_quant::GgufDequantError),
}

/// Convenient result alias for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;
