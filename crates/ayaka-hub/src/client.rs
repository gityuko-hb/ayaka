use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use ayaka_utils::progress_bar::{
    MultitaskProgress, ProgressBarColor, SpinnerBar, fmt_bytes, fmt_throughput,
    make_bytes_bar_style,
};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache as HfCache, Repo, RepoType};
use indicatif::ProgressBar;
use tokio::task::JoinSet;
use tracing::{info, trace, warn};

use crate::{
    cache::Cache,
    download::{DownloadProgress, download_file, download_url},
    error::{HubError, HubResult, classify_api_error},
    path::{
        hub_cache_dir, is_offline, offline_get, offline_snapshot_files, read_retry_base_delay_env,
        read_retry_max_env, read_token, snapshot_path,
    },
    retry::{RetryPolicy, backoff_for, is_retryable},
};

// ── Internal configuration (Send + Sync + Clone) ───────────────────────────────

/// Configuration stored in `HubClient`, cloneable into `spawn_blocking` tasks.
///
/// Holds everything needed to reconstruct a `hf_hub::api::sync::Api` inside a
/// blocking worker thread — avoiding the need to share a non-`Send` API handle.
#[derive(Clone, Debug)]
pub(crate) struct HubClientConfig {
    pub cache_dir: PathBuf,
    pub token: Option<String>,
    pub progress: bool,
    pub retry: RetryPolicy,
}

impl HubClientConfig {
    /// Build a fresh `hf_hub::api::sync::Api` from the stored config.
    ///
    /// Creating a new `Api` per task is the safest pattern: it avoids shared
    /// mutable state across threads and works without requiring `hf_hub::Api`
    /// to be `Send`.
    pub(crate) fn build_sync_api(&self) -> HubResult<hf_hub::api::sync::Api> {
        ApiBuilder::from_cache(HfCache::new(self.cache_dir.clone()))
            .with_progress(self.progress)
            .with_token(self.token.clone())
            .build()
            .map_err(|e| HubError::Api(e.to_string()))
    }
}

// ── ProgressHolder ─────────────────────────────────────────────────────────────

/// Shared progress context for `HubClient` operations.
///
/// By default, [`ProgressHolder::default()`] is silent — all operations run
/// without UI. Wrap a [`MultitaskProgress`](ayaka_utils::progress_bar::MultitaskProgress)
/// via [`from_multi_progress`](Self::from_multi_progress) to surface spinners
/// for list/get phases and a per-file progress bar for parallel shard fetches.
///
/// `ProgressHolder` is `Clone + Send + Sync` so it can be moved into
/// `spawn_blocking` tasks.
#[derive(Clone, Default)]
pub struct ProgressHolder {
    inner: Option<Arc<MultitaskProgress>>,
}

impl ProgressHolder {
    /// Silent (no UI) progress context. All operations run without bars.
    pub fn silent() -> Self {
        Self { inner: None }
    }

    /// Create a new internal `MultiProgress`. The caller may recover it later
    /// via [`into_multi_progress`](Self::into_multi_progress) to share with
    /// other components.
    pub fn enabled() -> Self {
        Self {
            inner: Some(Arc::new(MultitaskProgress::new())),
        }
    }

    /// Wrap a caller-owned `MultiProgress` so it can be driven by the
    /// `HubClient` while remaining shareable with other UI components.
    pub fn from_multi_progress(mp: indicatif::MultiProgress) -> Self {
        Self {
            inner: Some(Arc::new(MultitaskProgress::from_mp(mp))),
        }
    }

    /// `true` when at least one progress surface is active.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Consume `self` and return the inner `MultiProgress` so the caller can
    /// keep using it after the `HubClient` is dropped.
    pub fn into_multi_progress(self) -> Option<indicatif::MultiProgress> {
        self.inner.map(|arc| {
            match Arc::try_unwrap(arc) {
                Ok(tracker) => tracker.into_mp(),
                Err(_) => {
                    // Multiple references exist (e.g. HubClient clones held the
                    // holder); we cannot give the caller the original
                    // MultiProgress, so drop silently.
                    tracing::warn!(
                        "ProgressHolder::into_multi_progress: MultiProgress \
                         still shared; dropping the duplicated handle"
                    );
                    indicatif::MultiProgress::new()
                },
            }
        })
    }

    /// Add a per-file progress bar to the shared `MultiProgress`. Returns
    /// `None` when silent; the caller should skip progress updates in that
    /// case.
    pub(crate) fn add_task(
        &self,
        label: &str,
        total: u64,
        color: ProgressBarColor,
    ) -> Option<ProgressBar> {
        self.inner
            .as_ref()
            .map(|t| t.add_task(label, total, color))
    }

    /// Add a per-file byte-level progress bar (download / upload tracking).
    ///
    /// Uses the `{bytes}/{total_bytes} ({bytes_per_sec}, eta …)` template
    /// from [`make_bytes_bar_style`]. Returns `None` when silent; the caller
    /// should skip progress updates in that case.
    pub(crate) fn add_bytes_task(
        &self,
        label: &str,
        total: u64,
        color: ProgressBarColor,
    ) -> Option<ProgressBar> {
        let inner = self.inner.as_ref()?;
        let bar = ProgressBar::new(total);
        bar.set_style(make_bytes_bar_style(label, color));
        Some(inner.add_existing_bar(bar))
    }

    /// Add a spinner for an unknown-duration phase.
    pub(crate) fn add_spinner(
        &self,
        label: &'static str,
    ) -> Option<SpinnerBar> {
        self.inner.as_ref().map(|t| {
            // `MultitaskProgress` only exposes `add_task` / `add_spinner` (both
            // return `ProgressBar`). Wrap with a short-lived `SpinnerBar`
            // adapter for set_message ergonomics.
            let bar = t.add_spinner(label);
            SpinnerBar::from_indicatif(bar)
        })
    }

    /// Add a spinner with a runtime-formatted label (`String`).
    #[allow(dead_code)]
    pub(crate) fn add_spinner_label(
        &self,
        label: &str,
    ) -> Option<SpinnerBar> {
        self.inner.as_ref().map(|t| {
            let bar = t.add_spinner(label);
            SpinnerBar::from_indicatif(bar)
        })
    }
}

// ── LogPhase ──────────────────────────────────────────────────────────────────

/// RAII guard for a `tracing::info!` start/finish pair around a logical
/// phase. Drop logs the elapsed time; a panic in the guarded scope logs
/// `tracing::warn!` instead.
///
/// Visibility is governed by the caller's `RUST_LOG` — this guard never
/// forces a level.
pub(crate) struct LogPhase {
    name: &'static str,
    fields: String,
    started: Instant,
}

impl LogPhase {
    pub fn new(name: &'static str) -> Self {
        info!(phase = name, "starting phase");
        Self {
            name,
            fields: String::new(),
            started: Instant::now(),
        }
    }

    /// Attach a key=value fragment to the start log. Call before doing any
    /// non-trivial work so a partial failure still carries context.
    pub fn with_field(
        mut self,
        key: &'static str,
        value: impl std::fmt::Display,
    ) -> Self {
        if !self.fields.is_empty() {
            self.fields.push(' ');
        }
        self.fields.push_str(&format!("{key}={value}"));
        info!(phase = self.name, "{}", self.fields);
        self
    }
}

impl Drop for LogPhase {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        if std::thread::panicking() {
            warn!(
                phase = self.name,
                elapsed_ms = elapsed.as_millis() as u64,
                "phase aborted"
            );
        } else {
            info!(
                phase = self.name,
                elapsed_ms = elapsed.as_millis() as u64,
                "phase done"
            );
        }
    }
}

// ── HubClient ─────────────────────────────────────────────────────────────────

/// Async-facing HuggingFace Hub client.
///
/// All blocking network / filesystem operations are dispatched to
/// `tokio::task::spawn_blocking` so they never stall the async executor.
/// Internally uses the stable `hf_hub::api::sync` API wrapped with tokio.
///
/// # Example
/// ```no_run
/// # tokio_test::block_on(async {
/// use ayaka_hub::{HubClient, ModelLoadConfig, resolve_model_files};
///
/// let client = HubClient::from_env()?;
/// let config = ModelLoadConfig::new("mistralai/Mistral-7B-v0.1");
/// let paths = resolve_model_files(&client, &config).await?;
/// println!("tokenizer: {}", paths.tokenizer.display());
/// # Ok::<_, ayaka_hub::HubError>(())
/// # });
/// ```
pub struct HubClient {
    config: Arc<HubClientConfig>,
    cache: Cache,
    progress: ProgressHolder,
    /// Cached commit-hash lookups keyed by `(model_id, revision)`.  Populated
    /// lazily by [`HubClient::resolve_commit`]; lets the byte-level download
    /// path know where to write without re-fetching `info()` for every file.
    commit_cache: Arc<Mutex<HashMap<(String, String), String>>>,
}

impl HubClient {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Build a `HubClient` from environment variables (`HF_TOKEN`, `HF_HUB_CACHE`, …).
    pub fn from_env() -> HubResult<Self> {
        Self::builder().build()
    }

    /// Fine-grained builder.
    pub fn builder() -> HubClientBuilder {
        HubClientBuilder::default()
    }

    // ── Core API ──────────────────────────────────────────────────────────────

    /// List all files in a model repository.
    ///
    /// Resolution order:
    /// 1. Local path — if `model_id` points to an existing directory, list its
    ///    direct children (no network, no cache write).
    /// 2. File-list cache — return immediately when a valid (non-expired) entry
    ///    exists.
    /// 3. Offline snapshot — when `HF_HUB_OFFLINE` is set, walk the HF Hub
    ///    snapshot directory.  Populates the cache so the next call is fast.
    /// 4. Online — call `api.info()`, write result to cache, return.
    pub async fn list_files(
        &self,
        model_id: &str,
        revision: &str,
    ) -> HubResult<Vec<String>> {
        // ── Validate input ────────────────────────────────────────────────
        if model_id.is_empty() {
            return Err(HubError::Cache("model_id must not be empty".to_string()));
        }
        if revision.is_empty() {
            return Err(HubError::Cache(
                "revision must not be empty (use \"main\")".to_string(),
            ));
        }

        // ① Local directory.
        if Path::new(model_id).exists() {
            return list_local_dir(model_id);
        }

        // ② File-list cache (with TTL).
        if let Some(files) = self.cache.read(model_id) {
            return Ok(files);
        }

        // ③ Offline snapshot walk.
        if is_offline() {
            let spinner = self
                .progress
                .add_spinner("Listing repo (offline)");
            let cache_dir = match hub_cache_dir() {
                Some(d) => d,
                None => {
                    if let Some(sp) = spinner {
                        sp.finish_with_owned("No cache dir".to_string());
                    }
                    return Err(HubError::Cache(
                        "HF_HUB_OFFLINE is set but the HF cache directory \
                         cannot be determined. Set HF_HUB_CACHE or HF_HOME."
                            .to_string(),
                    ));
                },
            };
            let files = offline_snapshot_files(&cache_dir, model_id, revision);
            if !files.is_empty() {
                self.cache.write(model_id, &files);
                if let Some(sp) = spinner {
                    let count = files.len();
                    sp.finish_with_owned(format!("Found {count} files (offline)"));
                }
                return Ok(files);
            }
            if let Some(sp) = spinner {
                sp.finish_with_owned("No snapshot found".to_string());
            }
            return Err(HubError::Offline(format!(
                "HF_HUB_OFFLINE is set but no cached snapshot found for \
                 `{model_id}` (revision `{revision}`). \
                 Run `huggingface-cli download {model_id}` first."
            )));
        }

        // ④ Online: API info call in a blocking task.
        let _phase = LogPhase::new("list_files")
            .with_field("model_id", model_id)
            .with_field("revision", revision);
        let spinner = self.progress.add_spinner("Listing repo");
        let config = Arc::clone(&self.config);
        let model_id_owned = model_id.to_string();
        let revision_owned = revision.to_string();
        let retry = self.config.retry;

        let files: Vec<String> = tokio::task::spawn_blocking(move || {
            let mut attempt: u32 = 0;
            loop {
                let api = config.build_sync_api()?;
                let repo = api.repo(Repo::with_revision(
                    model_id_owned.clone(),
                    RepoType::Model,
                    revision_owned.clone(),
                ));
                match repo.info() {
                    Ok(info) => {
                        return Ok::<Vec<String>, HubError>(
                            info.siblings
                                .into_iter()
                                .map(|s| s.rfilename)
                                .collect(),
                        );
                    },
                    Err(e) => {
                        let hub_err = classify_api_error(&e);
                        if is_retryable(&hub_err) && attempt < retry.max_retries {
                            let delay = backoff_for(retry, attempt);
                            trace!(
                                "list_files attempt {} failed ({}); retrying in {:?}",
                                attempt, hub_err, delay
                            );
                            std::thread::sleep(delay);
                            attempt += 1;
                            continue;
                        }
                        return Err(hub_err);
                    },
                }
            }
        })
        .await
        .map_err(|e| HubError::Api(format!("list_files task panicked: {e}")))??;

        self.cache.write(model_id, &files);
        if let Some(sp) = spinner {
            let count = files.len();
            sp.finish_with_owned(format!("Listed {count} files"));
        }
        trace!("`{model_id}`: {} files fetched from API", files.len());
        Ok(files)
    }

    /// Resolve a single file, downloading from the Hub if necessary.
    ///
    /// Downloads use a reqwest-based transfer layer (see
    /// [`crate::download::download_file`]) so that per-chunk byte progress
    /// is visible when [`ProgressHolder::enabled`] is on.  The downloaded
    /// file is written to the standard HF snapshot path and is therefore
    /// readable by the offline walker on subsequent runs.
    pub async fn get_file(
        &self,
        model_id: &str,
        file: &str,
        revision: &str,
    ) -> HubResult<PathBuf> {
        // ── Validate input ────────────────────────────────────────────────
        if model_id.is_empty() {
            return Err(HubError::Cache("model_id must not be empty".to_string()));
        }
        if file.is_empty() {
            return Err(HubError::Cache("file must not be empty".to_string()));
        }
        if revision.is_empty() {
            return Err(HubError::Cache(
                "revision must not be empty (use \"main\")".to_string(),
            ));
        }

        // ① Local directory.
        let local = Path::new(model_id);
        if local.exists() {
            let path = local.join(file);
            if path.exists() {
                trace!("Local file `{file}` at `{}`", path.display());
                return Ok(path);
            }
            return Err(HubError::NotFound(format!(
                "`{file}` not found at local path `{}`",
                local.display()
            )));
        }

        // ② Offline cache.
        if is_offline() {
            let cache_dir = match hub_cache_dir() {
                Some(d) => d,
                None => {
                    return Err(HubError::Cache(
                        "HF_HUB_OFFLINE is set but the HF cache directory \
                         cannot be determined. Set HF_HUB_CACHE or HF_HOME."
                            .to_string(),
                    ));
                },
            };
            let path = offline_get(&cache_dir, model_id, file, revision).ok_or_else(|| {
                HubError::Offline(format!(
                    "HF_HUB_OFFLINE is set but `{file}` for `{model_id}` \
                     (revision `{revision}`) is not cached. \
                     Run `huggingface-cli download {model_id}` first."
                ))
            })?;
            info!(
                phase = "get_file",
                model_id = model_id,
                file = file,
                revision = revision,
                "resolved from offline cache"
            );
            return Ok(path);
        }

        // ③ Online: byte-level download via the reqwest transfer layer.
        let _phase = LogPhase::new("get_file")
            .with_field("model_id", model_id)
            .with_field("file", file)
            .with_field("revision", revision);

        let commit = self.resolve_commit(model_id, revision).await?;
        let cache_dir = self.config.cache_dir.clone();
        let dest = snapshot_path(&cache_dir, model_id, &commit, file);

        let bar = self
            .progress
            .add_bytes_task(file, 0, ProgressBarColor::Green);
        let progress = DownloadProgress { bar, total: 0 };
        let token = self.config.token.clone();
        let url = download_url(model_id, revision, file);

        let outcome = download_file(&url, token.as_deref(), &dest, progress).await?;

        // One-line human summary that mirrors the bar so piped output
        // (no TTY) still tells the user what happened.
        let bps = if outcome.elapsed.as_secs_f64() > 0.0 {
            outcome.bytes as f64 / outcome.elapsed.as_secs_f64()
        } else {
            0.0
        };
        if outcome.from_cache {
            info!(
                phase = "get_file",
                file = file,
                bytes = outcome.bytes,
                "served from cache ({})",
                fmt_bytes(outcome.bytes)
            );
        } else {
            info!(
                phase = "get_file",
                file = file,
                bytes = outcome.bytes,
                elapsed_ms = outcome.elapsed.as_millis() as u64,
                "downloaded {} in {:.2}s ({})",
                fmt_bytes(outcome.bytes),
                outcome.elapsed.as_secs_f64(),
                fmt_throughput(bps)
            );
        }

        Ok(dest)
    }

    /// Like [`get_file`] but returns `Ok(None)` for 404s instead of an error.
    ///
    /// Replaces mistral.rs's `try_get_file` which had a confusing
    /// `Result<Option<PathBuf>, ApiError>` return type and relied on implicit
    /// status-code matching.
    pub async fn try_get_file(
        &self,
        model_id: &str,
        file: &str,
        revision: &str,
    ) -> HubResult<Option<PathBuf>> {
        match self.get_file(model_id, file, revision).await {
            Ok(p) => Ok(Some(p)),
            Err(HubError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Resolve the commit hash for `(model_id, revision)`.
    ///
    /// Order:
    /// 1. In-memory cache (this is the common case — every file in a model
    ///    reuses one commit lookup).
    /// 2. The `refs/{revision}` file on disk when present (works offline).
    /// 3. `repo.info()` over the blocking hf-hub client.
    async fn resolve_commit(
        &self,
        model_id: &str,
        revision: &str,
    ) -> HubResult<String> {
        let key = (model_id.to_string(), revision.to_string());
        if let Some(c) = self
            .commit_cache
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
        {
            return Ok(c);
        }

        // ② Refs file (offline-friendly).
        let folder = format!("models--{}", model_id.replace('/', "--"));
        let ref_path = self
            .config
            .cache_dir
            .join(&folder)
            .join("refs")
            .join(revision);
        if let Ok(s) = std::fs::read_to_string(&ref_path) {
            let commit = s.trim().to_string();
            if !commit.is_empty() {
                self.commit_cache
                    .lock()
                    .unwrap()
                    .insert(key, commit.clone());
                return Ok(commit);
            }
        }

        // ③ API info() in a blocking task.
        let config = Arc::clone(&self.config);
        let model_id_owned = model_id.to_string();
        let revision_owned = revision.to_string();
        let retry = self.config.retry;

        let commit = tokio::task::spawn_blocking(move || -> HubResult<String> {
            let mut attempt: u32 = 0;
            loop {
                let api = config.build_sync_api()?;
                let repo = api.repo(Repo::with_revision(
                    model_id_owned.clone(),
                    RepoType::Model,
                    revision_owned.clone(),
                ));
                match repo.info() {
                    Ok(info) => return Ok(info.sha),
                    Err(e) => {
                        let hub_err = classify_api_error(&e);
                        if is_retryable(&hub_err) && attempt < retry.max_retries {
                            let delay = backoff_for(retry, attempt);
                            trace!(
                                "resolve_commit attempt {} failed ({}); retrying in {:?}",
                                attempt, hub_err, delay
                            );
                            std::thread::sleep(delay);
                            attempt += 1;
                            continue;
                        }
                        return Err(hub_err);
                    },
                }
            }
        })
        .await
        .map_err(|e| HubError::Api(format!("resolve_commit task panicked: {e}")))??;

        if commit.is_empty() {
            return Err(HubError::Api(format!(
                "`{model_id}` (revision `{revision}`): empty commit hash"
            )));
        }
        self.commit_cache
            .lock()
            .unwrap()
            .insert(key, commit.clone());
        Ok(commit)
    }

    /// Download `files` in parallel, returning resolved paths **in the same
    /// order as the input slice**.
    ///
    /// Downloads run through the reqwest-based transfer layer
    /// ([`crate::download::download_file`]) so per-chunk byte progress is
    /// visible when a [`ProgressHolder`] is enabled.  An overall count bar
    /// (`"Fetching N files"`) is kept on top of the per-file byte bars so
    /// a user can see "12/14 shards done" at a glance.
    ///
    /// For offline / local paths, falls back to sequential resolution
    /// (already cached, so the cost is negligible).
    ///
    /// This is the key performance fix over mistral.rs, where `get_paths!`
    /// fetches N shard files one after another (sequential blocking I/O).
    pub async fn parallel_fetch(
        &self,
        model_id: &str,
        files: &[&str],
        revision: &str,
    ) -> HubResult<Vec<PathBuf>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        // ── Validate input ────────────────────────────────────────────────
        if model_id.is_empty() {
            return Err(HubError::Cache("model_id must not be empty".to_string()));
        }
        if revision.is_empty() {
            return Err(HubError::Cache(
                "revision must not be empty (use \"main\")".to_string(),
            ));
        }

        // Offline / local paths are already cached; sequential resolution is fine.
        if is_offline() || Path::new(model_id).exists() {
            let mut paths = Vec::with_capacity(files.len());
            for &f in files {
                paths.push(self.get_file(model_id, f, revision).await?);
            }
            return Ok(paths);
        }

        let _phase = LogPhase::new("parallel_fetch")
            .with_field("model_id", model_id)
            .with_field("revision", revision)
            .with_field("count", files.len());

        // Resolve the commit once and compute all snapshot paths.
        let commit = self.resolve_commit(model_id, revision).await?;
        let cache_dir = self.config.cache_dir.clone();
        let dests: Vec<PathBuf> = files
            .iter()
            .map(|f| snapshot_path(&cache_dir, model_id, &commit, f))
            .collect();

        // Per-file byte bars (one row per file).
        let file_bars: Vec<Option<ProgressBar>> = files
            .iter()
            .map(|f| {
                self.progress
                    .add_bytes_task(f, 0, ProgressBarColor::Green)
            })
            .collect();

        // Overall count bar (kept per design — "12/14 shards done" at a
        // glance). Uses the count style, not byte style.
        let total = files.len() as u64;
        let overall_label = format!("Fetching {total} files");
        let overall_bar = self
            .progress
            .add_task(&overall_label, total, ProgressBarColor::Blue);

        let mut set: JoinSet<HubResult<(usize, PathBuf, u64, Duration)>> = JoinSet::new();
        let token = self.config.token.clone();

        for (idx, &file) in files.iter().enumerate() {
            let url = download_url(model_id, revision, file);
            let dest = dests[idx].clone();
            let file_bar = file_bars[idx].clone();
            let token = token.clone();

            set.spawn(async move {
                let progress = DownloadProgress {
                    bar: file_bar,
                    total: 0,
                };
                let result = download_file(&url, token.as_deref(), &dest, progress).await;
                match result {
                    Ok(o) => Ok((idx, dest, o.bytes, o.elapsed)),
                    Err(e) => Err(e),
                }
            });
        }

        let mut results: Vec<Option<PathBuf>> = vec![None; files.len()];
        let mut completed: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut total_elapsed: Duration = Duration::ZERO;

        while let Some(outcome) = set.join_next().await {
            match outcome {
                Err(join_err) => {
                    set.abort_all();
                    if let Some(ref bar) = overall_bar {
                        bar.finish_with_message("aborted");
                    }
                    return Err(HubError::Api(format!(
                        "parallel_fetch join error: {join_err}"
                    )));
                },
                Ok(Err(hub_err)) => {
                    set.abort_all();
                    if let Some(ref bar) = overall_bar {
                        bar.finish_with_message("aborted");
                    }
                    return Err(hub_err);
                },
                Ok(Ok((idx, path, bytes, elapsed))) => {
                    results[idx] = Some(path);
                    completed += 1;
                    total_bytes = total_bytes.saturating_add(bytes);
                    total_elapsed += elapsed;
                    if let Some(ref bar) = overall_bar {
                        bar.set_position(completed);
                    }
                },
            }
        }

        if let Some(ref bar) = overall_bar {
            bar.finish_with_message(format!("Fetched {completed}/{total}"));
        }

        // One-line summary so piped output (no TTY) still tells the story.
        let avg_bps = if total_elapsed.as_secs_f64() > 0.0 {
            total_bytes as f64 / total_elapsed.as_secs_f64()
        } else {
            0.0
        };
        info!(
            phase = "parallel_fetch",
            model_id = model_id,
            revision = revision,
            files = total,
            total_bytes = total_bytes,
            elapsed_ms = total_elapsed.as_millis() as u64,
            "downloaded {} across {} files in {:.2}s ({})",
            fmt_bytes(total_bytes),
            total,
            total_elapsed.as_secs_f64(),
            fmt_throughput(avg_bps)
        );

        // All results present — the Option<> layer is a safety net for the
        // index-keyed array; it should never be None here.
        Ok(results.into_iter().flatten().collect())
    }

    /// Force-invalidate the file-list cache entry for `model_id`.
    pub fn invalidate_cache(
        &self,
        model_id: &str,
    ) {
        self.cache.invalidate(model_id);
    }
}

// ── Builder ────────────────────────────────────────────────────────────────────

/// Fine-grained builder for [`HubClient`].
#[derive(Default)]
pub struct HubClientBuilder {
    token: Option<String>,
    cache_dir: Option<PathBuf>,
    progress: bool,
    ttl_secs: Option<u64>,
    progress_holder: Option<ProgressHolder>,
    max_retries: Option<u32>,
    retry_base_delay_ms: Option<u64>,
}

impl HubClientBuilder {
    pub fn with_token(
        mut self,
        token: impl Into<String>,
    ) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn with_cache_dir(
        mut self,
        dir: PathBuf,
    ) -> Self {
        self.cache_dir = Some(dir);
        self
    }

    /// Show download progress bars (default: `false`).
    pub fn with_progress(
        mut self,
        progress: bool,
    ) -> Self {
        self.progress = progress;
        self
    }

    /// Override the file-list cache TTL (seconds, default: 3600).
    pub fn with_ttl(
        mut self,
        secs: u64,
    ) -> Self {
        self.ttl_secs = Some(secs);
        self
    }

    /// Provide a [`ProgressHolder`] for surfacing progress bars during
    /// `list_files`, `get_file`, and `parallel_fetch`. Default: silent.
    pub fn with_progress_holder(
        mut self,
        holder: ProgressHolder,
    ) -> Self {
        self.progress_holder = Some(holder);
        self
    }

    /// Set the maximum number of retry attempts after the first failure
    /// (default: 0 — no retry). Retries trigger on transient errors: connection
    /// reset / abort / refused, timeout, broken pipe, HTTP 5xx, and HTTP 429.
    /// Auth, NotFound, Offline, and Cache errors are never retried.
    ///
    /// Resolution order: explicit builder value → `AYAKA_HUB_MAX_RETRIES` env
    /// var → default (0).
    pub fn with_max_retries(
        mut self,
        n: u32,
    ) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Set the base delay (milliseconds) for exponential backoff between
    /// retries (default: 100). Delay after attempt `n` is
    /// `base_delay_ms * 2^n`, capped at 5 seconds.
    ///
    /// Resolution order: explicit builder value → `AYAKA_HUB_RETRY_BASE_DELAY_MS`
    /// env var → default (100).
    pub fn with_retry_base_delay_ms(
        mut self,
        ms: u64,
    ) -> Self {
        self.retry_base_delay_ms = Some(ms);
        self
    }

    pub fn build(self) -> HubResult<HubClient> {
        let cache_dir = self
            .cache_dir
            .or_else(hub_cache_dir)
            .ok_or_else(|| HubError::Cache("Cannot determine HF cache directory".into()))?;

        std::fs::create_dir_all(&cache_dir)?;

        let token = self.token.or_else(read_token);

        // Retry resolution: builder > env > default.
        let defaults = RetryPolicy::default();
        let max_retries = self
            .max_retries
            .or_else(read_retry_max_env)
            .unwrap_or(defaults.max_retries);
        let base_delay_ms = self
            .retry_base_delay_ms
            .or_else(read_retry_base_delay_env)
            .unwrap_or(defaults.base_delay_ms);
        let retry = RetryPolicy {
            max_retries,
            base_delay_ms,
        };

        let inner_config = Arc::new(HubClientConfig {
            cache_dir: cache_dir.clone(),
            token,
            progress: self.progress,
            retry,
        });

        let mut cache = Cache::new(cache_dir);
        if let Some(ttl) = self.ttl_secs {
            cache = cache.with_ttl(ttl);
        }

        Ok(HubClient {
            config: inner_config,
            cache,
            progress: self.progress_holder.unwrap_or_default(),
            commit_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// List the immediate children of a local directory.
fn list_local_dir(dir: &str) -> HubResult<Vec<String>> {
    let entries = std::fs::read_dir(dir)?;
    Ok(entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(String::from)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    // ── LogPhase RAII guard ─────────────────────────────────────────────

    /// `LogPhase::drop` should not panic on normal drop.
    #[test]
    fn log_phase_does_not_panic_on_normal_drop() {
        let _p = LogPhase::new("unit-test-phase");
        // drop runs here; we just need to confirm no panic.
    }

    /// `LogPhase::drop` should not panic when the thread is panicking,
    /// so a mid-phase failure still tears down the guard cleanly.
    #[test]
    fn log_phase_does_not_panic_when_thread_panics() {
        let result = std::panic::catch_unwind(|| {
            let _p = LogPhase::new("unit-test-panic-phase");
            panic!("forced panic");
        });
        assert!(result.is_err(), "panic should propagate");
    }

    /// `with_field` should record the key=value fragment for the start log.
    #[test]
    fn log_phase_with_field_attaches_metadata() {
        let _p = LogPhase::new("unit-test-fields").with_field("model_id", "org/m");
        // no panic = the formatting path is sound
    }

    fn test_client() -> (TempDir, HubClient) {
        let dir = TempDir::new().unwrap();
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .build()
            .unwrap();
        (dir, client)
    }

    #[tokio::test]
    async fn list_files_rejects_empty_model_id() {
        let (_dir, client) = test_client();
        let err = client
            .list_files("", "main")
            .await
            .expect_err("empty model_id must error");
        assert!(matches!(err, HubError::Cache(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn list_files_rejects_empty_revision() {
        let (_dir, client) = test_client();
        let err = client
            .list_files("org/model", "")
            .await
            .expect_err("empty revision must error");
        assert!(matches!(err, HubError::Cache(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn get_file_rejects_empty_file() {
        let (_dir, client) = test_client();
        let err = client
            .get_file("org/model", "", "main")
            .await
            .expect_err("empty file must error");
        assert!(matches!(err, HubError::Cache(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn parallel_fetch_rejects_empty_model_id() {
        let (_dir, client) = test_client();
        let err = client
            .parallel_fetch("", &["config.json"], "main")
            .await
            .expect_err("empty model_id must error");
        assert!(matches!(err, HubError::Cache(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn progress_holder_default_is_silent() {
        let h = ProgressHolder::default();
        assert!(!h.is_enabled());
        assert!(h.into_multi_progress().is_none());
    }

    #[tokio::test]
    async fn progress_holder_enabled_round_trip() {
        let h = ProgressHolder::enabled();
        assert!(h.is_enabled());
        // Should be the only Arc reference, so `try_unwrap` succeeds.
        let mp = h.into_multi_progress();
        assert!(mp.is_some());
    }

    #[test]
    #[serial(hub_env)]
    fn builder_default_retry_is_zero() {
        let dir = TempDir::new().unwrap();
        // Make sure no env var leaks in from another test / shell.
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_MAX_ENV) };
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_BASE_DELAY_ENV) };
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .build()
            .unwrap();
        assert_eq!(client.config.retry.max_retries, 0);
        assert_eq!(client.config.retry.base_delay_ms, 100);
    }

    #[test]
    #[serial(hub_env)]
    fn builder_with_max_retries_wins() {
        let dir = TempDir::new().unwrap();
        unsafe { std::env::set_var(crate::path::HUB_RETRY_MAX_ENV, "7") };
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .with_max_retries(2)
            .build()
            .unwrap();
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_MAX_ENV) };
        // Builder explicit value beats env var.
        assert_eq!(client.config.retry.max_retries, 2);
    }

    #[test]
    #[serial(hub_env)]
    fn env_var_used_when_builder_omits() {
        let dir = TempDir::new().unwrap();
        unsafe { std::env::set_var(crate::path::HUB_RETRY_MAX_ENV, "5") };
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .build()
            .unwrap();
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_MAX_ENV) };
        assert_eq!(client.config.retry.max_retries, 5);
    }

    #[test]
    #[serial(hub_env)]
    fn invalid_env_falls_back_to_default() {
        let dir = TempDir::new().unwrap();
        unsafe { std::env::set_var(crate::path::HUB_RETRY_MAX_ENV, "garbage") };
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .build()
            .unwrap();
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_MAX_ENV) };
        assert_eq!(client.config.retry.max_retries, 0);
    }

    #[test]
    #[serial(hub_env)]
    fn base_delay_resolution_priority() {
        let dir = TempDir::new().unwrap();
        unsafe {
            std::env::set_var(crate::path::HUB_RETRY_BASE_DELAY_ENV, "999");
        }
        let client = HubClient::builder()
            .with_cache_dir(dir.path().to_path_buf())
            .with_retry_base_delay_ms(50)
            .build()
            .unwrap();
        unsafe { std::env::remove_var(crate::path::HUB_RETRY_BASE_DELAY_ENV) };
        assert_eq!(client.config.retry.base_delay_ms, 50);
    }
}
