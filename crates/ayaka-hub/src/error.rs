use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    /// HTTP 401 or 403 — authentication / authorization failure.
    #[error("Authentication failed (HTTP {0}). Set HF_TOKEN or run `helix login`.")]
    Auth(u16),

    /// HTTP 404 — the requested model or file does not exist (or is private).
    #[error("Not found (HTTP 404): {0}")]
    NotFound(String),

    /// Other non-success HTTP responses.
    #[error("HTTP error {code}: {message}")]
    Http { code: u16, message: String },

    /// `HF_HUB_OFFLINE` is set but the requested resource is not in the local cache.
    #[error("Offline mode (HF_HUB_OFFLINE is set): {0}")]
    Offline(String),

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Cache serialization / deserialization error.
    #[error("Cache error: {0}")]
    Cache(String),

    /// Any other error from the upstream hf-hub crate.
    #[error("HuggingFace Hub API error: {0}")]
    Api(String),
}

pub type HubResult<T> = Result<T, HubError>;

// ── Conversions ────────────────────────────────────────────────────────────────

impl From<serde_json::Error> for HubError {
    fn from(e: serde_json::Error) -> Self {
        HubError::Cache(e.to_string())
    }
}

/// Classify a raw `hf-hub` [`ApiError`] into a typed [`HubError`].
///
/// Attempts to match on `TooManyRetries` first (recursive), then falls back to
/// scanning the error message for an HTTP status code. This is kept in a single
/// centralised function so that when `hf-hub` exposes `status_code()` natively
/// we only need to update one place.
///
/// TODO: upstream issue <https://github.com/huggingface/hf-hub> to expose
///       `fn status_code(&self) -> Option<u16>` on `ApiError`.
pub(crate) fn classify_api_error(err: &hf_hub::api::sync::ApiError) -> HubError {
    let code = extract_status_code(err);
    match code {
        Some(c @ (401 | 403)) => HubError::Auth(c),
        Some(404) => HubError::NotFound(err.to_string()),
        Some(c) => HubError::Http {
            code: c,
            message: err.to_string(),
        },
        None => HubError::Api(err.to_string()),
    }
}

fn extract_status_code(err: &hf_hub::api::sync::ApiError) -> Option<u16> {
    use hf_hub::api::sync::ApiError;

    // Recurse through retry wrappers.
    if let ApiError::TooManyRetries(inner) = err {
        return extract_status_code(inner);
    }

    // Fallback: scan the error message.  Handles both "status code N" and "HTTP N".
    let msg = err.to_string();
    for marker in ["status code ", "HTTP "] {
        if let Some((_, tail)) = msg.split_once(marker)
            && let Ok(n) = tail
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u16>()
        {
            return Some(n);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_error_display() {
        let e = HubError::Auth(401);
        assert!(e.to_string().contains("401"));

        let e = HubError::NotFound("Qwen/Qwen3-1.7B".into());
        assert!(e.to_string().contains("404"));

        let e = HubError::Http {
            code: 500,
            message: "server error".into(),
        };
        assert!(e.to_string().contains("500"));
    }
}
