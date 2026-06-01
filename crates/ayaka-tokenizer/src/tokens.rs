use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    Literal(String),
    EnvVar(String),
    Path(PathBuf),
    CacheToken,
    None,
}

pub fn get_token(source: &TokenSource) -> Option<String> {
    match source {
        TokenSource::Literal(token) => normalize_token(token.clone()),
        TokenSource::EnvVar(name) => std::env::var(name).ok().and_then(normalize_token),
        TokenSource::Path(path) => std::fs::read_to_string(path)
            .ok()
            .and_then(normalize_token),
        TokenSource::CacheToken => cache_token(),
        TokenSource::None => None,
    }
}

#[cfg(feature = "hub-download")]
pub(crate) fn resolve_default_token(explicit: Option<&str>) -> Option<String> {
    explicit
        .and_then(|token| normalize_token(token.to_owned()))
        .or_else(|| get_token(&TokenSource::EnvVar("HF_TOKEN".into())))
        .or_else(|| get_token(&TokenSource::EnvVar("HF_HUB_TOKEN".into())))
        .or_else(|| get_token(&TokenSource::CacheToken))
}

fn cache_token() -> Option<String> {
    let path = dirs::home_dir()?
        .join(".cache")
        .join("huggingface")
        .join("token");
    std::fs::read_to_string(path)
        .ok()
        .and_then(normalize_token)
}

fn normalize_token(token: String) -> Option<String> {
    let trimmed = token.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}
