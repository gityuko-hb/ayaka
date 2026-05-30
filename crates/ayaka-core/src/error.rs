#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {msg}, location: {location}")]
    Internal { msg: String, location: String },

    #[error("OOM: requested {requested_mb}MB, available {available_mb}MB")]
    OutOfMemory {
        requested_mb: usize,
        available_mb: usize,
    },

    #[error("OOM Device: device {device}, needed: {needed_mb}MB, available: {available_mb}MB")]
    OutOfMemoryDevice {
        device: String,
        needed_mb: usize,
        available_mb: usize,
    },

    #[error("Model load failed: {path}")]
    LoadFailure {
        path: std::path::PathBuf,
        #[source]
        cause: anyhow::Error,
    },

    #[error("Tokenization error: {0}")]
    TokenizationError(#[from] tokenizers::Error),

    #[error("Context length exceeded: {requested} > {max}")]
    ContextLengthExceeded { requested: usize, max: usize },

    #[error("Unsupported Device: {0}")]
    UnsupportedDevice(String),
}
