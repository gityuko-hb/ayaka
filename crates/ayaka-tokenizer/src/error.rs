use std::path::PathBuf;

pub type TokenizerResult<T> = Result<T, TokenizerError>;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Tokenizer format not recognized in: {0}")]
    UnknownFormat(PathBuf),
    #[error(
        "HuggingFace token not found (tried HF_TOKEN, HF_HUB_TOKEN, ~/.cache/huggingface/token)"
    )]
    TokenNotFound,
    #[error("Failed to download from HF Hub: {0}")]
    HubError(String),
    #[error("Backend error: {0}")]
    BackendError(String),
    #[error("Chat template not available for this model")]
    NoChatTemplate,
    #[error("Chat template render failed: {0}")]
    TemplateRenderError(String),
    #[error("Invalid UTF-8 in detokenizer buffer")]
    InvalidUtf8,
    #[error("Token ID {0} out of vocabulary range")]
    OutOfVocab(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl TokenizerError {
    #[allow(dead_code)]
    pub(crate) fn backend(error: impl Into<String>) -> Self {
        Self::BackendError(error.into())
    }
}
