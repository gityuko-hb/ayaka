//! Byte-level downloader used by `HubClient`.
//!
//! The high-level `hf_hub::api::sync::Api::Repo::get` path runs over `ureq`
//! and exposes no per-chunk callback, so it cannot drive a byte-level progress
//! bar. This module bypasses `hf-hub` for the byte transfer while reusing its
//! cache and snapshot layout: callers compute the snapshot path via
//! [`crate::path::snapshot_path`], then call [`download_file`] which streams
//! the response body chunk-by-chunk into a tokio file, calling
//! [`indicatif::ProgressBar::set_position`] on every chunk.

use std::path::Path;
use std::time::Instant;

use futures_util::StreamExt;
use indicatif::ProgressBar;
use tokio::io::AsyncWriteExt;
use tracing::{debug, trace};

use crate::error::{HubError, HubResult};

/// Bundle carried by [`download_file`] so callers can drive their own
/// progress surface (a byte-style bar when `total > 0`, a spinner otherwise).
pub(crate) struct DownloadProgress {
    pub bar: Option<ProgressBar>,
    #[allow(dead_code)]
    pub total: u64,
}

/// Outcome of a single download — used by callers to build a one-line
/// `tracing::info!` summary after the bar has been finalised.
pub(crate) struct DownloadOutcome {
    pub bytes: u64,
    pub elapsed: std::time::Duration,
    pub from_cache: bool,
}

/// Build the HuggingFace download URL for a single file inside a repo.
pub(crate) fn download_url(
    model_id: &str,
    revision: &str,
    file: &str,
) -> String {
    format!("https://huggingface.co/{model_id}/resolve/{revision}/{file}")
}

/// Download `url` to `dest` with per-chunk progress updates.
///
/// The destination is created (truncated) if absent.  If the file already
/// exists and its size matches `Content-Length`, the function returns
/// immediately with `from_cache = true` and never opens the network.  If the
/// server omits `Content-Length` (e.g. chunked transfer without a hint) the
/// bar is left without a `total` and behaves like a spinner.
pub(crate) async fn download_file(
    url: &str,
    token: Option<&str>,
    dest: &Path,
    mut progress: DownloadProgress,
) -> HubResult<DownloadOutcome> {
    let started = Instant::now();

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Build a fresh client per call — reqwest::Client is cheap and avoids
    // shared mutable state across concurrent downloads.
    let client = reqwest::Client::builder()
        .user_agent(concat!("ayaka-hub/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| HubError::Api(format!("reqwest build: {e}")))?;

    // HEAD probe to learn Content-Length and to validate the URL before we
    // open a file handle.
    let mut head_req = client.head(url);
    if let Some(t) = token {
        head_req = head_req.bearer_auth(t);
    }
    let head_resp = head_req
        .send()
        .await
        .map_err(|e| HubError::Api(format!("HEAD {url}: {e}")))?;
    let head_status = head_resp.status();
    if !head_status.is_success() {
        return Err(classify_head_status(head_status, url));
    }
    let total = head_resp.content_length().unwrap_or(0);
    drop(head_resp);

    // Cache hit: file already present and matches Content-Length.
    if total > 0 {
        match tokio::fs::metadata(dest).await {
            Ok(m) if m.len() == total => {
                if let Some(bar) = progress.bar.take() {
                    bar.finish_and_clear();
                }
                return Ok(DownloadOutcome {
                    bytes: total,
                    elapsed: started.elapsed(),
                    from_cache: true,
                });
            },
            Ok(_) => {
                trace!(
                    "`{}` exists but size mismatch (have {}, want {}); re-downloading",
                    dest.display(),
                    tokio::fs::metadata(dest)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0),
                    total
                );
            },
            Err(_) => {},
        }
    }

    if let Some(bar) = progress.bar.as_ref() {
        if total > 0 {
            bar.set_length(total);
        } else {
            // Unknown total — indicatif will show speed but not a percentage.
            bar.set_length(0);
        }
    }

    // Real GET.
    let mut get_req = client.get(url);
    if let Some(t) = token {
        get_req = get_req.bearer_auth(t);
    }
    let resp = get_req
        .send()
        .await
        .map_err(|e| HubError::Api(format!("GET {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        return Err(match code {
            401 | 403 => HubError::Auth(code),
            404 => HubError::NotFound(format!("GET {url}")),
            _ => HubError::Http {
                code,
                message: format!("GET {url}: {status}"),
            },
        });
    }
    let content_length = resp.content_length().unwrap_or(total);
    if content_length != total && total == 0 {
        // Server omitted Content-Length in HEAD; capture the value now.
        if let Some(bar) = progress.bar.as_ref() {
            bar.set_length(content_length);
        }
    }

    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(dest).await?;
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| HubError::Api(format!("read chunk: {e}")))?;
        file.write_all(&chunk).await?;
        written = written.saturating_add(chunk.len() as u64);
        if let Some(bar) = progress.bar.as_ref() {
            bar.set_position(written);
        }
    }
    file.flush().await?;
    file.sync_all().await?;

    debug!(
        url = url,
        bytes = written,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "download complete"
    );

    Ok(DownloadOutcome {
        bytes: written,
        elapsed: started.elapsed(),
        from_cache: false,
    })
}

fn classify_head_status(
    status: reqwest::StatusCode,
    url: &str,
) -> HubError {
    let code = status.as_u16();
    match code {
        401 | 403 => HubError::Auth(code),
        404 => HubError::NotFound(format!("HEAD {url}")),
        _ => HubError::Http {
            code,
            message: format!("HEAD {url}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_url_uses_resolve_segment() {
        let u = download_url("org/model", "main", "config.json");
        assert_eq!(
            u,
            "https://huggingface.co/org/model/resolve/main/config.json"
        );
    }

    #[test]
    fn download_url_encodes_nested_org() {
        let u = download_url("meta-llama/Llama-3-8B", "v1.0", "model.safetensors");
        assert_eq!(
            u,
            "https://huggingface.co/meta-llama/Llama-3-8B/resolve/v1.0/model.safetensors"
        );
    }

    #[test]
    fn silent_progress_has_no_bar() {
        let p = DownloadProgress {
            bar: None,
            total: 0,
        };
        assert!(p.bar.is_none());
        assert_eq!(p.total, 0);
    }

    #[test]
    fn classify_head_status_maps_known_codes() {
        assert!(matches!(
            classify_head_status(reqwest::StatusCode::UNAUTHORIZED, "u"),
            HubError::Auth(401)
        ));
        assert!(matches!(
            classify_head_status(reqwest::StatusCode::FORBIDDEN, "u"),
            HubError::Auth(403)
        ));
        assert!(matches!(
            classify_head_status(reqwest::StatusCode::NOT_FOUND, "u"),
            HubError::NotFound(_)
        ));
        assert!(matches!(
            classify_head_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "u"),
            HubError::Http { code: 500, .. }
        ));
    }
}
