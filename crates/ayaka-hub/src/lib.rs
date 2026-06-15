//! # ayaka-hub
//!
//! HuggingFace Hub integration for the Helix LLM inference engine.
//!
//! ## Quick start
//!
//! ```no_run
//! # tokio_test::block_on(async {
//! use ayaka_hub::{HubClient, ModelLoadConfig, resolve_model_files};
//!
//! let client = HubClient::from_env()?;
//! let config = ModelLoadConfig::new("qwen/Qwen3-1.7B").with_revision("v1.0.0");
//! let paths  = resolve_model_files(&client, &config).await?;
//!
//! println!("tokenizer : {}", paths.tokenizer.display());
//! println!("config    : {}", paths.config.display());
//! println!("shards    : {}", paths.weights.len());
//! # Ok::<_, ayaka_hub::HubError>(())
//! # });
//! ```

pub mod cache;
pub mod client;
pub mod error;
pub mod path;
pub mod resolver;
pub mod retry;

// ── Flat re-exports (primary public API) ──────────────────────────────────────

pub use client::{HubClient, HubClientBuilder, ProgressHolder};
pub use error::{HubError, HubResult};
pub use path::{
    HUB_OFFLINE_ENV, HUB_RETRY_BASE_DELAY_ENV, HUB_RETRY_MAX_ENV, hub_cache_dir, is_offline,
    read_retry_base_delay_env, read_retry_max_env, read_token,
};
pub use resolver::{
    ModelLoadConfig, ModelPaths, resolve_model_files, resolve_quantized_files, resolve_uqff_shards,
};
pub use retry::{RetryPolicy, backoff_for, is_retryable};
