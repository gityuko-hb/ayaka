//! Architecture-agnostic model metadata, parsed defensively from `config.json`.

use serde::Deserialize;

use crate::error::{LoaderError, Result};

/// Normalized description of a decoder-only transformer, covering dense-GQA,
/// MoE, and MLA (DeepSeek-V3) variants.
///
/// Fields are populated from a Hugging Face `config.json` (or inferred from
/// GGUF metadata). Derived geometry (`head_dim`, `num_kv_heads`) is exposed
/// through methods so callers never re-derive it inconsistently.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    /// Canonical architecture id used for registry lookup (e.g. `"qwen3"`).
    pub arch_id: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// `None` => same as `num_attention_heads` (no GQA).
    pub num_key_value_heads: Option<usize>,
    /// `None` => derived as `hidden_size / num_attention_heads`.
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// MLA / MoE fields (DeepSeek-V3-style); absent for dense models.
    pub mla: Option<MlaConfig>,
    pub moe: Option<MoeConfig>,
}

/// Multi-head latent attention parameters (DeepSeek-V3).
#[derive(Debug, Clone, PartialEq)]
pub struct MlaConfig {
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub q_lora_rank: Option<usize>,
}

/// Mixture-of-experts parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeConfig {
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub n_shared_experts: usize,
    /// Number of leading layers that stay dense before MoE layers begin.
    pub first_k_dense_replace: usize,
}

impl ModelMetadata {
    /// Per-head dimension. Independent of `hidden_size / num_heads` when the
    /// config supplies an explicit `head_dim` (e.g. Qwen3).
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Number of KV heads, falling back to attention heads when not GQA.
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    /// Parse from a `config.json` string.
    pub fn from_config_json(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json).map_err(|source| LoaderError::Parse {
            what: "config.json".to_string(),
            source,
        })?;
        raw.into_metadata()
    }
}

/// Raw `config.json` shape. Every numeric field is optional so a missing key
/// produces a clear `InvalidConfig` error rather than a serde failure.
#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,

    // MLA
    kv_lora_rank: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    q_lora_rank: Option<usize>,

    // MoE
    n_routed_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    n_shared_experts: usize,
    #[serde(default)]
    first_k_dense_replace: usize,
}

fn default_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    4096
}

impl RawConfig {
    fn into_metadata(self) -> Result<ModelMetadata> {
        let arch_id = self
            .model_type
            .clone()
            .or_else(|| {
                self.architectures
                    .first()
                    .map(|a| normalize_arch(a))
            })
            .ok_or_else(|| {
                LoaderError::InvalidConfig("missing both model_type and architectures".to_string())
            })?;

        let require = |v: Option<usize>, name: &str| -> Result<usize> {
            v.ok_or_else(|| LoaderError::InvalidConfig(format!("missing required field `{name}`")))
        };

        let mla = match (self.kv_lora_rank, self.qk_rope_head_dim) {
            (Some(kv_lora_rank), Some(qk_rope_head_dim)) => Some(MlaConfig {
                kv_lora_rank,
                qk_rope_head_dim,
                qk_nope_head_dim: self.qk_nope_head_dim.unwrap_or(0),
                q_lora_rank: self.q_lora_rank,
            }),
            _ => None,
        };

        let moe = match (self.n_routed_experts, self.num_experts_per_tok) {
            (Some(n_routed_experts), Some(num_experts_per_tok)) => Some(MoeConfig {
                n_routed_experts,
                num_experts_per_tok,
                moe_intermediate_size: self
                    .moe_intermediate_size
                    .unwrap_or(self.intermediate_size.unwrap_or(0)),
                n_shared_experts: self.n_shared_experts,
                first_k_dense_replace: self.first_k_dense_replace,
            }),
            _ => None,
        };

        Ok(ModelMetadata {
            arch_id,
            hidden_size: require(self.hidden_size, "hidden_size")?,
            intermediate_size: require(self.intermediate_size, "intermediate_size")?,
            num_hidden_layers: require(self.num_hidden_layers, "num_hidden_layers")?,
            num_attention_heads: require(self.num_attention_heads, "num_attention_heads")?,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            vocab_size: require(self.vocab_size, "vocab_size")?,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings,
            mla,
            moe,
        })
    }
}

/// Map a HF `architectures` entry (e.g. `"Qwen3ForCausalLM"`) to a lowercase id.
fn normalize_arch(arch: &str) -> String {
    let trimmed = arch
        .strip_suffix("ForCausalLM")
        .or_else(|| arch.strip_suffix("Model"))
        .unwrap_or(arch);
    trimmed.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3: &str = r#"{
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": true
    }"#;

    #[test]
    fn parses_qwen3_with_explicit_head_dim() {
        let m = ModelMetadata::from_config_json(QWEN3).unwrap();
        assert_eq!(m.arch_id, "qwen3");
        // head_dim is explicit and NOT hidden/heads (1024/16 = 64).
        assert_eq!(m.head_dim(), 128);
        assert_eq!(m.num_kv_heads(), 8);
        assert!(m.tie_word_embeddings);
        assert!(m.mla.is_none());
        assert!(m.moe.is_none());
    }

    #[test]
    fn derives_head_dim_and_kv_heads_when_absent() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 512,
            "intermediate_size": 1376,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "vocab_size": 32000
        }"#;
        let m = ModelMetadata::from_config_json(json).unwrap();
        assert_eq!(m.arch_id, "llama");
        assert_eq!(m.head_dim(), 64); // 512 / 8
        assert_eq!(m.num_kv_heads(), 8); // falls back to attn heads
    }

    #[test]
    fn parses_deepseek_mla_and_moe() {
        let json = r#"{
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "vocab_size": 129280,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "first_k_dense_replace": 3
        }"#;
        let m = ModelMetadata::from_config_json(json).unwrap();
        let mla = m.mla.as_ref().unwrap();
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_rope_head_dim, 64);
        let moe = m.moe.as_ref().unwrap();
        assert_eq!(moe.n_routed_experts, 256);
        assert_eq!(moe.first_k_dense_replace, 3);
    }

    #[test]
    fn missing_required_field_is_clear_error() {
        let json = r#"{ "model_type": "qwen3", "hidden_size": 1024 }"#;
        let err = ModelMetadata::from_config_json(json).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }
}
