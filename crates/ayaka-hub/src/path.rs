use std::{
    env, fs,
    path::{Path, PathBuf},
};

use tracing::{trace, warn};

pub const HUB_OFFLINE_ENV: &str = "HF_HUB_OFFLINE";
pub const HUB_CACHE_ENV: &str = "HF_HUB_CACHE";
pub const HUB_HOME_ENV: &str = "HF_HOME";
pub const HUB_TOKEN_ENV: &str = "HF_TOKEN";
pub const HUB_RETRY_MAX_ENV: &str = "AYAKA_HUB_MAX_RETRIES";
pub const HUB_RETRY_BASE_DELAY_ENV: &str = "AYAKA_HUB_RETRY_BASE_DELAY_MS";

/// Returns `true` when the user has requested fully-offline operation via
/// `HF_HUB_OFFLINE`.
///
/// Accepted truthy values: `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive).
/// Anything else, or an unset variable, is treated as online.
pub fn is_offline() -> bool {
    matches!(
        env::var(HUB_OFFLINE_ENV)
            .ok()
            .as_deref()
            .map(|v| v.trim().to_ascii_lowercase()),
        Some(ref v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
    )
}

// Directory resolution
/// Resolve the HuggingFace hub cache directory.
///
/// Precedence:
/// 1. `HF_HUB_CACHE`
/// 2. `HF_HOME/hub`
/// 3. `~/.cache/huggingface/hub`
pub fn hub_cache_dir() -> Option<PathBuf> {
    let dir = env::var(HUB_CACHE_ENV)
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            env::var(HUB_HOME_ENV)
                .ok()
                .map(|h| PathBuf::from(h).join("hub"))
        })
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub")));

    if let Some(ref d) = dir
        && let Err(e) = fs::create_dir_all(d)
    {
        warn!("Cannot create hub cache directory `{}`: {e}", d.display());
    }

    dir
}

/// Resolve the HuggingFace token file path (`HF_HOME/token` or
/// `~/.cache/huggingface/token`).
pub fn token_path() -> Option<PathBuf> {
    env::var(HUB_HOME_ENV)
        .ok()
        .map(|h| PathBuf::from(h).join("token"))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("token")))
}

// ── Token loading ──────────────────────────────────────────────────────────────

/// Read the HF authentication token from the environment or the token file.
///
/// Precedence: `HF_TOKEN` env var → `~/.cache/huggingface/token`.
pub fn read_token() -> Option<String> {
    env::var(HUB_TOKEN_ENV)
        .ok()
        .filter(|t| !t.is_empty())
        .or_else(|| {
            token_path()
                .and_then(|p| fs::read_to_string(p).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

// ── Retry env resolution ───────────────────────────────────────────────────────

/// Parse the value of `AYAKA_HUB_MAX_RETRIES` (default: `None` ⇒ builder
/// default 0).
///
/// `None` or an empty / whitespace string yields `None`.  Invalid values are
/// logged as warnings and treated as unset so a typo never silently changes
/// behavior.
pub fn parse_retry_max(raw: Option<&str>) -> Option<u32> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<u32>() {
        Ok(n) => Some(n),
        Err(e) => {
            warn!("{HUB_RETRY_MAX_ENV}={s:?} is not a valid u32 ({e}); ignoring");
            None
        },
    }
}

/// Read `AYAKA_HUB_MAX_RETRIES` (default: `None` ⇒ builder default 0).
pub fn read_retry_max_env() -> Option<u32> {
    parse_retry_max(env::var(HUB_RETRY_MAX_ENV).ok().as_deref())
}

/// Parse the value of `AYAKA_HUB_RETRY_BASE_DELAY_MS` (default: `None` ⇒
/// builder default 100).
pub fn parse_retry_base_delay(raw: Option<&str>) -> Option<u64> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<u64>() {
        Ok(n) => Some(n),
        Err(e) => {
            warn!("{HUB_RETRY_BASE_DELAY_ENV}={s:?} is not a valid u64 ({e}); ignoring");
            None
        },
    }
}

/// Read `AYAKA_HUB_RETRY_BASE_DELAY_MS` (default: `None` ⇒ builder default 100).
pub fn read_retry_base_delay_env() -> Option<u64> {
    parse_retry_base_delay(env::var(HUB_RETRY_BASE_DELAY_ENV).ok().as_deref())
}

// ── Snapshot resolution ────────────────────────────────────────────────────────

/// HF cache folder name for a given model id.
///
/// The HF Hub convention is `models--{org}--{repo}` (slashes replaced with `--`).
fn snapshot_folder(model_id: &str) -> String {
    format!("models--{}", model_id.replace('/', "--"))
}

/// Resolve the commit hash for `revision` from the refs file, falling back to
/// treating `revision` itself as a commit hash when the file is absent.
///
/// Returns an empty string when `revision` is empty so callers can detect
/// invalid input via the resulting empty snapshot directory.
fn resolve_commit(
    cache_dir: &Path,
    folder: &str,
    revision: &str,
) -> String {
    if revision.is_empty() {
        return String::new();
    }
    let ref_path = cache_dir.join(folder).join("refs").join(revision);
    fs::read_to_string(&ref_path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| revision.to_string())
}

/// Look up a single file in the local HF snapshot cache.
///
/// Returns `Some(path)` when the file is present in the snapshot directory,
/// `None` otherwise.  This is intentionally **non-panicking** — mistral.rs's
/// equivalent could panic if `unwrap()`-ing a missing optional path.
pub fn offline_get(
    cache_dir: &Path,
    model_id: &str,
    file: &str,
    revision: &str,
) -> Option<PathBuf> {
    let folder = snapshot_folder(model_id);
    let commit = resolve_commit(cache_dir, &folder, revision);

    let path = cache_dir
        .join(&folder)
        .join("snapshots")
        .join(&commit)
        .join(file);

    if path.exists() {
        trace!(
            "Offline cache hit: `{file}` for `{model_id}` at `{}`",
            path.display()
        );
        Some(path)
    } else {
        None
    }
}

/// Return all file paths under the snapshot directory for `(model_id, revision)`,
/// relative to the snapshot root.
///
/// Returns an empty `Vec` (never panics) when the snapshot directory does not
/// exist or the walk fails.
pub fn offline_snapshot_files(
    cache_dir: &Path,
    model_id: &str,
    revision: &str,
) -> Vec<String> {
    fn walk(
        root: &Path,
        dir: &Path,
        out: &mut Vec<String>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if let Ok(rel) = path.strip_prefix(root) {
                // Normalise path separators (important on Windows).
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }

    let folder = snapshot_folder(model_id);
    let commit = resolve_commit(cache_dir, &folder, revision);
    let snapshot_dir = cache_dir
        .join(&folder)
        .join("snapshots")
        .join(&commit);

    if !snapshot_dir.is_dir() {
        return Vec::new();
    }

    let mut files = Vec::new();
    if let Err(e) = walk(&snapshot_dir, &snapshot_dir, &mut files) {
        warn!("Failed to walk snapshot dir for `{model_id}`: {e}");
    }

    trace!(
        "Offline snapshot for `{model_id}` (rev `{revision}`): {} files",
        files.len()
    );
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    // ── is_offline: reads HUB_OFFLINE_ENV, so all these tests race on
    // the same process-global variable.  Marked `#[serial]` so they
    // run one at a time and never see each other's values.
    #[test]
    #[serial]
    fn is_offline_truthy_values() {
        for val in ["1", "true", "True", "TRUE", "yes", "on"] {
            unsafe { std::env::set_var(HUB_OFFLINE_ENV, val) };
            assert!(is_offline(), "`{val}` should be truthy");
        }
        unsafe { std::env::remove_var(HUB_OFFLINE_ENV) };
    }

    #[test]
    #[serial]
    fn is_offline_falsy_values() {
        for val in ["0", "false", "no", "off", ""] {
            unsafe { std::env::set_var(HUB_OFFLINE_ENV, val) };
            assert!(!is_offline(), "`{val}` should be falsy");
        }
        unsafe { std::env::remove_var(HUB_OFFLINE_ENV) };
    }

    #[test]
    #[serial]
    fn is_offline_unset_is_online() {
        unsafe { std::env::remove_var(HUB_OFFLINE_ENV) };
        assert!(!is_offline());
    }

    // ── parse_retry_max: pure function, no env mutation needed. ─────
    #[test]
    fn parse_retry_max_parses_valid_value() {
        assert_eq!(parse_retry_max(Some("3")), Some(3));
    }

    #[test]
    fn parse_retry_max_returns_none_when_unset() {
        assert_eq!(parse_retry_max(None), None);
    }

    #[test]
    fn parse_retry_max_ignores_invalid_value() {
        assert_eq!(parse_retry_max(Some("not-a-number")), None);
    }

    #[test]
    fn parse_retry_max_ignores_empty_value() {
        assert_eq!(parse_retry_max(Some("")), None);
    }

    #[test]
    fn parse_retry_max_trims_whitespace() {
        assert_eq!(parse_retry_max(Some("  7  ")), Some(7));
    }

    // ── parse_retry_base_delay: pure function, no env mutation. ────
    #[test]
    fn parse_retry_base_delay_parses_valid_value() {
        assert_eq!(parse_retry_base_delay(Some("250")), Some(250));
    }

    #[test]
    fn parse_retry_base_delay_ignores_invalid_value() {
        assert_eq!(parse_retry_base_delay(Some("abc")), None);
    }

    #[test]
    fn parse_retry_base_delay_trims_whitespace() {
        assert_eq!(parse_retry_base_delay(Some("  42\n")), Some(42));
    }

    // ── Env→read wiring: a single serial test per variable verifies
    // that read_*_env() reads the correct env-var name.  These are
    // the only tests that actually need to touch the process env.
    #[test]
    #[serial]
    fn read_retry_max_env_reads_ayaka_hub_max_retries() {
        unsafe { std::env::set_var(HUB_RETRY_MAX_ENV, "4") };
        assert_eq!(read_retry_max_env(), Some(4));
        unsafe { std::env::remove_var(HUB_RETRY_MAX_ENV) };
    }

    #[test]
    #[serial]
    fn read_retry_base_delay_env_reads_ayaka_hub_retry_base_delay_ms() {
        unsafe { std::env::set_var(HUB_RETRY_BASE_DELAY_ENV, "250") };
        assert_eq!(read_retry_base_delay_env(), Some(250));
        unsafe { std::env::remove_var(HUB_RETRY_BASE_DELAY_ENV) };
    }

    #[test]
    fn snapshot_folder_replaces_slash() {
        assert_eq!(snapshot_folder("org/model"), "models--org--model");
        assert_eq!(
            snapshot_folder("meta-llama/Llama-3-8B"),
            "models--meta-llama--Llama-3-8B"
        );
    }

    #[test]
    fn offline_get_finds_file() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create the HF cache layout:
        //   models--org--repo/refs/main  →  "abc123"
        //   models--org--repo/snapshots/abc123/model.safetensors
        let folder = "models--org--repo";
        let commit = "abc123";
        let refs_dir = root.join(folder).join("refs");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), commit).unwrap();

        let snap_dir = root.join(folder).join("snapshots").join(commit);
        fs::create_dir_all(&snap_dir).unwrap();
        fs::write(snap_dir.join("model.safetensors"), b"").unwrap();

        let result = offline_get(root, "org/repo", "model.safetensors", "main");
        assert!(result.is_some());
    }

    #[test]
    fn offline_get_missing_returns_none() {
        let dir = TempDir::new().unwrap();
        let result = offline_get(dir.path(), "org/repo", "missing.bin", "main");
        assert!(result.is_none());
    }

    #[test]
    fn offline_snapshot_files_lists_all() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let folder = "models--org--m";
        let commit = "deadbeef";
        let refs_dir = root.join(folder).join("refs");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), commit).unwrap();

        let snap_dir = root.join(folder).join("snapshots").join(commit);
        fs::create_dir_all(&snap_dir).unwrap();
        for name in [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            fs::write(snap_dir.join(name), b"").unwrap();
        }

        let files = offline_snapshot_files(root, "org/m", "main");
        assert_eq!(files.len(), 3);
        assert!(files.contains(&"config.json".to_string()));
    }

    #[test]
    fn offline_snapshot_files_empty_when_missing() {
        let dir = TempDir::new().unwrap();
        let files = offline_snapshot_files(dir.path(), "nonexistent/model", "main");
        assert!(files.is_empty());
    }
}
