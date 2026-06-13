use std::{collections::HashSet, path::PathBuf};

use tracing::{trace, warn};

use crate::{
    client::HubClient,
    error::{HubError, HubResult},
};

/// Configuration for resolving all model file paths.
///
/// This is the single struct that replaces the 8-parameter `get_paths!` macro
/// call-site in mistral.rs.  Using a struct makes fields optional, self-
/// documenting, and trivially extensible without breaking callers.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig<'a> {
    /// HuggingFace model ID (e.g. `"qwen/Qwen3-1.7B"`) or a local
    /// directory path.
    pub model_id: &'a str,

    /// Revision / branch / commit hash (default: `"main"`).
    pub revision: &'a str,

    /// If `Some`, use this local path as the tokenizer file instead of
    /// downloading it from the Hub.
    pub tokenizer_path: Option<&'a str>,
}

impl<'a> ModelLoadConfig<'a> {
    pub fn new(model_id: &'a str) -> Self {
        Self {
            model_id,
            revision: "main",
            tokenizer_path: None,
        }
    }

    pub fn with_revision(
        mut self,
        revision: &'a str,
    ) -> Self {
        self.revision = revision;
        self
    }

    pub fn with_tokenizer(
        mut self,
        path: &'a str,
    ) -> Self {
        self.tokenizer_path = Some(path);
        self
    }
}

// ── Resolved paths ─────────────────────────────────────────────────────────────

/// All resolved file paths for a model — the output of [`resolve_model_files`].
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Tokenizer vocabulary file (`tokenizer.json` or `tekken.json`).
    pub tokenizer: PathBuf,

    /// Model architecture config (`config.json`).
    pub config: PathBuf,

    /// Weight shard files (`.safetensors`).  Order mirrors the repo listing.
    pub weights: Vec<PathBuf>,

    /// `generation_config.json`, if present.
    pub generation_config: Option<PathBuf>,

    /// `tokenizer_config.json`, if present.
    pub chat_template: Option<PathBuf>,

    /// `preprocessor_config.json`, if present (multimodal models).
    pub preprocessor_config: Option<PathBuf>,

    /// `processor_config.json`, if present.
    pub processor_config: Option<PathBuf>,
}

// ── Main resolver ──────────────────────────────────────────────────────────────

/// Build an O(1) lookup set from a `dir_list` so per-name checks
/// (`contains`) cost O(1) instead of the O(N) `Vec::iter().any(...)` pattern.
fn build_lookup(dir_list: &[String]) -> HashSet<&str> {
    dir_list.iter().map(String::as_str).collect()
}

/// Resolve all model file paths, downloading from HF Hub if necessary.
///
/// This single async function replaces the entire `get_paths!` /
/// `get_embedding_paths!` / `get_paths_gguf!` macro family from mistral.rs,
/// together with the `api_dir_list!` and `api_get_file!` helper macros.
pub async fn resolve_model_files(
    client: &HubClient,
    config: &ModelLoadConfig<'_>,
) -> HubResult<ModelPaths> {
    let model_id = config.model_id;
    let revision = config.revision;

    // ── Step 1: list repo ────────────────────────────────────────────────────
    let dir_list = client.list_files(model_id, revision).await?;
    let lookup = build_lookup(&dir_list);
    trace!(
        "Resolving model files for `{model_id}` (revision `{revision}`): {} entries",
        dir_list.len()
    );

    // ── Step 2: tokenizer ────────────────────────────────────────────────────
    let tokenizer = if let Some(local) = config.tokenizer_path {
        trace!("Using provided tokenizer: `{local}`");
        PathBuf::from(local)
    } else if lookup.contains("tokenizer.json") {
        client
            .get_file(model_id, "tokenizer.json", revision)
            .await?
    } else if lookup.contains("tekken.json") {
        client
            .get_file(model_id, "tekken.json", revision)
            .await?
    } else {
        // Unlike mistral.rs (which silently traces and then fails with a
        // confusing 404), we emit a clear warning here and return a typed error.
        warn!(
            "`{model_id}`: neither `tokenizer.json` nor `tekken.json` found in \
             repo file listing. The model may not have been fully uploaded."
        );
        return Err(HubError::NotFound(format!(
            "No tokenizer file (`tokenizer.json` or `tekken.json`) found in `{model_id}`"
        )));
    };

    // ── Step 3: config.json ──────────────────────────────────────────────────
    let config_path = client
        .get_file(model_id, "config.json", revision)
        .await?;

    // ── Step 4: weight shards (parallel download) ────────────────────────────
    let shard_names: Vec<&str> = dir_list
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .map(String::as_str)
        .collect();

    if shard_names.is_empty() {
        return Err(HubError::NotFound(format!(
            "No `.safetensors` files found in `{model_id}`. \
             For GGUF / UQFF formats use `resolve_quantized_files` instead."
        )));
    }

    let weights = client
        .parallel_fetch(model_id, &shard_names, revision)
        .await?;

    trace!("`{model_id}`: resolved {} weight shard(s)", weights.len());

    // ── Step 5: optional files ───────────────────────────────────────────────
    let generation_config = maybe_get(
        client,
        model_id,
        "generation_config.json",
        revision,
        &lookup,
    )
    .await?;
    let chat_template =
        maybe_get(client, model_id, "tokenizer_config.json", revision, &lookup).await?;
    let preprocessor_config = maybe_get(
        client,
        model_id,
        "preprocessor_config.json",
        revision,
        &lookup,
    )
    .await?;
    let processor_config =
        maybe_get(client, model_id, "processor_config.json", revision, &lookup).await?;

    Ok(ModelPaths {
        tokenizer,
        config: config_path,
        weights,
        generation_config,
        chat_template,
        preprocessor_config,
        processor_config,
    })
}

// Quantized resolver
/// Resolve paths for a GGUF or UQFF quantized model.
///
/// Downloads the quantized weight file from `(quant_model_id, quant_filename)`
/// while still fetching config / tokenizer from the base `model_id`.
///
/// Returns `(ModelPaths, quant_path)` where `ModelPaths.weights` is empty
/// (quantized weights live at `quant_path`).
pub async fn resolve_quantized_files(
    client: &HubClient,
    base: &ModelLoadConfig<'_>,
    quant_model_id: &str,
    quant_filename: &str,
) -> HubResult<(ModelPaths, PathBuf)> {
    let base_model_id = base.model_id;
    let revision = base.revision;

    // Fetch the quantized weight file.
    let quant_path = client
        .get_file(quant_model_id, quant_filename, revision)
        .await?;

    // Config and tokenizer always come from the base model.
    let dir_list = client.list_files(base_model_id, revision).await?;
    let lookup = build_lookup(&dir_list);

    let tokenizer = if let Some(local) = base.tokenizer_path {
        PathBuf::from(local)
    } else if lookup.contains("tokenizer.json") {
        client
            .get_file(base_model_id, "tokenizer.json", revision)
            .await?
    } else if lookup.contains("tekken.json") {
        client
            .get_file(base_model_id, "tekken.json", revision)
            .await?
    } else {
        return Err(HubError::NotFound(format!(
            "No tokenizer found in base model `{base_model_id}`"
        )));
    };

    let config_path = client
        .get_file(base_model_id, "config.json", revision)
        .await?;
    let generation_config = maybe_get(
        client,
        base_model_id,
        "generation_config.json",
        revision,
        &lookup,
    )
    .await?;
    let chat_template = maybe_get(
        client,
        base_model_id,
        "tokenizer_config.json",
        revision,
        &lookup,
    )
    .await?;

    let model_paths = ModelPaths {
        tokenizer,
        config: config_path,
        weights: Vec::new(), // quantized weights are at `quant_path`
        generation_config,
        chat_template,
        preprocessor_config: None,
        processor_config: None,
    };

    Ok((model_paths, quant_path))
}

// ── UQFF shard auto-discovery ─────────────────────────────────────────────────
/// Discover and resolve UQFF shard files from a repository.
///
/// Filters for files matching `<stem>-<N>-of-<M>.uqff` and fetches them all
/// in parallel.  This replaces the `get_uqff_paths!` macro's shard auto-
/// discovery loop (which called `hf::list_repo_files` directly, bypassing
/// `api_dir_list!`, and silently swallowed errors via `.unwrap_or_default()`).
///
/// `stem` is a filename prefix, e.g. "mistral-7b-instruct-q4k", which
/// matches files like `mistral-7b-instruct-q4k-00001-of-00003.uqff`.
pub async fn resolve_uqff_shards(
    client: &HubClient,
    model_id: &str,
    revision: &str,
    stem: &str,
) -> HubResult<Vec<PathBuf>> {
    let dir_list = client.list_files(model_id, revision).await?;

    let shard_names: Vec<&str> = dir_list
        .iter()
        .filter(|f| f.ends_with(".uqff") && f.starts_with(stem))
        .map(String::as_str)
        .collect();

    if shard_names.is_empty() {
        return Err(HubError::NotFound(format!(
            "No UQFF shards matching `{stem}*.uqff` found in `{model_id}`"
        )));
    }

    client
        .parallel_fetch(model_id, &shard_names, revision)
        .await
}

/// Fetch `filename` only when it appears in `lookup`; return `Ok(None)`
/// otherwise.
///
/// This single generic helper replaces the 8+ repetitions of the pattern:
/// ```rust,ignore
/// let field = if dir_list.contains(&"X".to_string()) {
///     Some(api_get_file!(api, "X", model_id, &revision))
/// } else { None };
/// ```
///
/// `lookup` is the pre-built `HashSet<&str>` from `build_lookup(&dir_list)`,
/// giving O(1) membership checks instead of the original O(N) `iter().any()`.
pub(crate) async fn maybe_get(
    client: &HubClient,
    model_id: &str,
    filename: &str,
    revision: &str,
    lookup: &HashSet<&str>,
) -> HubResult<Option<PathBuf>> {
    if lookup.contains(filename) {
        Ok(Some(
            client
                .get_file(model_id, filename, revision)
                .await?,
        ))
    } else {
        Ok(None)
    }
}

/// Try multiple candidate filenames, returning the first one found.
#[allow(dead_code)]
pub(crate) async fn maybe_get_multi(
    client: &HubClient,
    model_id: &str,
    revision: &str,
    lookup: &HashSet<&str>,
    candidates: &[&str],
) -> HubResult<Option<PathBuf>> {
    for &name in candidates {
        if let Some(path) = maybe_get(client, model_id, name, revision, lookup).await? {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the config builder API is ergonomic.
    #[test]
    fn model_load_config_builder() {
        let c = ModelLoadConfig::new("org/model")
            .with_revision("v1.0")
            .with_tokenizer("/tmp/tok.json");

        assert_eq!(c.model_id, "org/model");
        assert_eq!(c.revision, "v1.0");
        assert_eq!(c.tokenizer_path, Some("/tmp/tok.json"));
    }

    #[test]
    fn model_load_config_defaults() {
        let c = ModelLoadConfig::new("org/model");
        assert_eq!(c.revision, "main");
        assert!(c.tokenizer_path.is_none());
    }

    #[test]
    fn build_lookup_o1_membership() {
        let dir_list = vec![
            "config.json".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
            "tokenizer.json".to_string(),
        ];
        let lookup = build_lookup(&dir_list);
        assert_eq!(lookup.len(), 4);
        assert!(lookup.contains("config.json"));
        assert!(lookup.contains("tokenizer.json"));
        assert!(!lookup.contains("missing.bin"));
    }

    #[test]
    fn build_lookup_empty() {
        let dir_list: Vec<String> = vec![];
        let lookup = build_lookup(&dir_list);
        assert!(lookup.is_empty());
    }
}
