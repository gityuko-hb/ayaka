use std::{
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fd_lock::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::error::HubResult;

/// Default time-to-live for cached repo file lists (1 hour).
pub const DEFAULT_TTL_SECS: u64 = 3_600;

// On-disk format
/// Serialized form of a repo file-list cache entry.
///
/// Adding `cached_at_secs` is the key difference from mistral.rs's `FileListCache`,
/// which had no TTL field and therefore returned stale listings forever.
#[derive(Serialize, Deserialize)]
pub struct FileListCache {
    pub files: Vec<String>,
    /// Unix timestamp (seconds) at which this entry was written.
    pub cached_at_secs: u64,
}

/// Manages the JSON file-list cache stored inside the HF hub cache directory.
#[derive(Clone, Debug)]
pub struct Cache {
    cache_dir: PathBuf,
    ttl_secs: u64,
}

impl Cache {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            ttl_secs: DEFAULT_TTL_SECS,
        }
    }

    /// Override the time-to-live (builder pattern).
    pub fn with_ttl(
        mut self,
        ttl_secs: u64,
    ) -> Self {
        self.ttl_secs = ttl_secs;
        self
    }

    fn cache_path(
        &self,
        model_id: &str,
    ) -> PathBuf {
        // Naming convention (stable since the first public release):
        //
        //   "org/model"      → "org-model_file_list.json"
        //   "org/model/"     → "org-model_file_list.json"   (trailing / stripped)
        //
        // Do **not** confuse this with:
        //   • `_repo_list.json`  — a mistral.rs legacy name that we deliberately
        //     did NOT adopt. The file we write is `<stem>_file_list.json`.
        //   • `models--{org}--{model}/` — the snapshot directory managed by
        //     `hf-hub` itself; it lives in the same `cache_dir` parent and
        //     holds the actual downloaded weight/config/tokenizer files. We
        //     only read from it (via `offline_*` helpers) and write our
        //     `*_file_list.json` next to it.
        let stem = model_id.trim_end_matches('/').replace('/', "-");
        self.cache_dir
            .join(format!("{stem}_file_list.json"))
    }

    /// Sidecar lock file used to coordinate multi-process access. Lives next
    /// to the cache file so the OS guarantees the lock and the cache are on
    /// the same filesystem (required for atomic `rename`).
    fn lock_path(
        &self,
        model_id: &str,
    ) -> PathBuf {
        self.cache_path(model_id)
            .with_extension("json.lock")
    }

    /// Open the lock file for `path`, creating it if absent. Returns a
    /// `RwLock` ready to acquire a read or write guard.
    fn open_lock(path: &Path) -> std::io::Result<RwLock<fs::File>> {
        let f = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        Ok(RwLock::new(f))
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Try to read a valid (non-expired) cache entry.
    ///
    /// Returns `None` if the file is absent, unreadable, unparseable, corrupt,
    /// or older than [`ttl_secs`](Self::with_ttl).
    ///
    /// Holds an advisory **read** lock on the sidecar `.json.lock` file for
    /// the duration of the read so that a concurrent `write` cannot rename
    /// the file out from under us. The lock is advisory; processes that do
    /// not participate in `ayaka-hub` are not blocked.
    pub fn read(
        &self,
        model_id: &str,
    ) -> Option<Vec<String>> {
        let path = self.cache_path(model_id);

        if !path.exists() {
            return None;
        }

        // Acquire advisory read lock — short-lived; released on drop.
        let lock_path = self.lock_path(model_id);
        let lock_file = match Self::open_lock(&lock_path) {
            Ok(l) => l,
            Err(e) => {
                warn!("Cannot open lock `{}`: {e}", lock_path.display());
                return None;
            },
        };
        let _guard = match lock_file.read() {
            Ok(g) => g,
            Err(e) => {
                warn!("Cannot acquire read lock on `{}`: {e}", lock_path.display());
                return None;
            },
        };

        let file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Cannot open file-list cache `{}`: {e}", path.display());
                return None;
            },
        };

        // Zero-copy streaming parse — avoids the double allocation in mistral.rs's
        // `read_to_string` + `from_str` pattern.
        let entry: FileListCache = match serde_json::from_reader(std::io::BufReader::new(file)) {
            Ok(e) => e,
            Err(e) => {
                warn!("Corrupt file-list cache `{}`: {e}", path.display());
                // Remove the corrupt file so the next call triggers a fresh fetch.
                let _ = fs::remove_file(&path);
                return None;
            },
        };

        let age = Self::now_secs().saturating_sub(entry.cached_at_secs);
        if age > self.ttl_secs {
            trace!(
                "File-list cache expired for `{model_id}` (age={age}s > ttl={}s)",
                self.ttl_secs
            );
            return None;
        }

        trace!(
            "File-list cache hit for `{model_id}` ({} files, age={age}s)",
            entry.files.len()
        );
        Some(entry.files)
    }

    /// Write a file list to cache using an **atomic rename**.
    ///
    /// Writes to `<path>.tmp` first, then calls `fs::rename()`.  On POSIX
    /// systems `rename()` is atomic within the same filesystem, preventing
    /// the race condition where another process reads a half-written file.
    ///
    /// Errors are logged as warnings but are **non-fatal** — a write failure
    /// means the next call will just refetch from the API.
    pub fn write(
        &self,
        model_id: &str,
        files: &[String],
    ) {
        if let Err(e) = self.write_inner(model_id, files) {
            warn!("Cannot write file-list cache for `{model_id}`: {e}");
        } else {
            trace!(
                "File-list cache written for `{model_id}` ({} files)",
                files.len()
            );
        }
    }

    fn write_inner(
        &self,
        model_id: &str,
        files: &[String],
    ) -> HubResult<()> {
        let path = self.cache_path(model_id);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Acquire advisory **write** lock — blocks other writers and readers
        // for the same `model_id` so the tmp+rename pair cannot race with
        // a concurrent read.
        let lock_path = self.lock_path(model_id);
        let mut lock_file = Self::open_lock(&lock_path)?;
        let _guard = lock_file.write().map_err(|e| {
            crate::error::HubError::Cache(format!(
                "Cannot acquire write lock on `{}`: {e}",
                lock_path.display()
            ))
        })?;

        let entry = FileListCache {
            files: files.to_vec(),
            cached_at_secs: Self::now_secs(),
        };

        // Write to a temp file alongside the target.
        let tmp = path.with_extension("json.tmp");
        {
            let f = fs::File::create(&tmp)?;
            // Compact JSON (not pretty-printed) — this is a machine-read cache.
            serde_json::to_writer(BufWriter::new(f), &entry)?;
        }

        // Atomic promotion: rename is POSIX-atomic on the same filesystem.
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }

        Ok(())
    }

    /// Force invalidate the cache entry for `model_id`.
    ///
    /// The next call to [`read`](Self::read) will return `None`, triggering a
    /// fresh API fetch.  Useful after manually adding files to a repo.
    ///
    /// Acquires the same write lock as [`write`](Self::write) so the
    /// invalidation cannot race with an in-progress write.
    pub fn invalidate(
        &self,
        model_id: &str,
    ) {
        let path = self.cache_path(model_id);
        if !path.exists() {
            return;
        }
        let lock_path = self.lock_path(model_id);
        let mut lock_file = match Self::open_lock(&lock_path) {
            Ok(l) => l,
            Err(e) => {
                warn!("Cannot open lock `{}`: {e}", lock_path.display());
                return;
            },
        };
        let _guard = match lock_file.write() {
            Ok(g) => g,
            Err(e) => {
                warn!(
                    "Cannot acquire write lock for invalidation `{}`: {e}",
                    lock_path.display()
                );
                return;
            },
        };
        if let Err(e) = fs::remove_file(&path) {
            warn!("Cannot invalidate file-list cache for `{model_id}`: {e}");
        } else {
            trace!("File-list cache invalidated for `{model_id}`");
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_cache(ttl: u64) -> (TempDir, Cache) {
        let dir = TempDir::new().unwrap();
        let cache = Cache::new(dir.path().to_path_buf()).with_ttl(ttl);
        (dir, cache)
    }

    #[test]
    fn round_trip() {
        let (_dir, cache) = test_cache(60);
        let files = vec!["model.safetensors".to_string(), "config.json".to_string()];

        cache.write("mistralai/Mistral-7B", &files);
        let got = cache.read("mistralai/Mistral-7B").unwrap();
        assert_eq!(got, files);
    }

    #[test]
    fn miss_when_absent() {
        let (_dir, cache) = test_cache(60);
        assert!(cache.read("some/model").is_none());
    }

    #[test]
    fn expired_returns_none() {
        let (_dir, cache) = test_cache(0); // TTL = 0s → always expired
        let files = vec!["a.safetensors".to_string()];
        cache.write("org/model", &files);
        // Ensure we cross a second boundary so `age > ttl` is true even
        // when write and read happen within the same second on fast hardware.
        std::thread::sleep(std::time::Duration::from_millis(1_100));
        assert!(cache.read("org/model").is_none());
    }

    #[test]
    fn cache_path_sanitises_slash() {
        let dir = TempDir::new().unwrap();
        let cache = Cache::new(dir.path().to_path_buf());
        let p = cache.cache_path("org/model-name");
        assert!(p.to_str().unwrap().contains("org-model-name"));
    }

    #[test]
    fn cache_path_strips_trailing_slash() {
        let dir = TempDir::new().unwrap();
        let cache = Cache::new(dir.path().to_path_buf());
        let p1 = cache.cache_path("org/model");
        let p2 = cache.cache_path("org/model/");
        let p3 = cache.cache_path("org/model///");
        assert_eq!(p1, p2, "trailing slash should not affect the path");
        assert_eq!(p2, p3, "multiple trailing slashes should all be stripped");
    }

    #[test]
    fn cache_path_ends_with_file_list_json() {
        let dir = TempDir::new().unwrap();
        let cache = Cache::new(dir.path().to_path_buf());
        let p = cache.cache_path("mistralai/Mistral-7B-v0.1");
        assert!(
            p.to_str().unwrap().ends_with("_file_list.json"),
            "must use *_file_list.json naming (NOT _repo_list.json): {p:?}"
        );
        assert!(
            !p.to_str().unwrap().contains("_repo_list"),
            "must not contain _repo_list: {p:?}"
        );
    }

    #[test]
    fn atomic_write_produces_no_tmp_on_success() {
        let (_dir, cache) = test_cache(3600);
        cache.write("org/m", &["f.safetensors".to_string()]);

        let path = cache.cache_path("org/m");
        let tmp = path.with_extension("json.tmp");
        assert!(path.exists(), "final cache file should exist");
        assert!(!tmp.exists(), "tmp file should have been renamed away");
    }

    #[test]
    fn invalidate_removes_entry() {
        let (_dir, cache) = test_cache(3600);
        cache.write("org/m", &["x".to_string()]);
        assert!(cache.read("org/m").is_some());

        cache.invalidate("org/m");
        assert!(cache.read("org/m").is_none());
    }

    #[test]
    fn corrupt_cache_returns_none_and_cleans_up() {
        let dir = TempDir::new().unwrap();
        let cache = Cache::new(dir.path().to_path_buf());
        let path = cache.cache_path("org/m");
        fs::write(&path, b"{{not valid json}}").unwrap();

        assert!(cache.read("org/m").is_none());
        assert!(!path.exists(), "corrupt cache file should be removed");
    }

    #[test]
    fn lock_sidecar_is_created_on_write() {
        let (_dir, cache) = test_cache(60);
        cache.write("org/lock-test", &["a.safetensors".to_string()]);
        let lock_path = cache.lock_path("org/lock-test");
        assert!(
            lock_path.exists(),
            "lock sidecar should be created during write"
        );
    }

    #[test]
    fn concurrent_writers_do_not_corrupt() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().unwrap();
        let cache = Arc::new(Cache::new(dir.path().to_path_buf()).with_ttl(3600));
        let model_id = "org/concurrent";
        let mut handles = Vec::new();

        for i in 0..4 {
            let cache = Arc::clone(&cache);
            let payload = vec![format!("file-{i}.safetensors")];
            handles.push(thread::spawn(move || {
                cache.write(model_id, &payload);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // The final state must be one of the 4 valid payloads — never partial
        // or garbled.
        let got = cache.read(model_id).expect("read after writes");
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with("file-"));
        assert!(got[0].ends_with(".safetensors"));
    }
}
