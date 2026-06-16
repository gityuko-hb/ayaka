//! Qwen3 configuration, derived from architecture-agnostic [`ModelMetadata`].

use ayaka_loader::ModelMetadata;

/// Concrete Qwen3 hyperparameters.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Project the shared metadata onto Qwen3's fields.
    pub fn from_metadata(m: &ModelMetadata) -> Self {
        Self {
            hidden_size: m.hidden_size,
            intermediate_size: m.intermediate_size,
            num_layers: m.num_hidden_layers,
            num_heads: m.num_attention_heads,
            num_kv_heads: m.num_kv_heads(),
            head_dim: m.head_dim(),
            vocab_size: m.vocab_size,
            rms_eps: m.rms_norm_eps,
            rope_theta: m.rope_theta,
            max_position_embeddings: m.max_position_embeddings,
            tie_word_embeddings: m.tie_word_embeddings,
        }
    }

    /// GQA repeat factor.
    pub fn n_rep(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Parameter count of one decoder layer (used for ledger accounting).
    pub fn layer_param_count(&self) -> usize {
        let hd = self.head_dim;
        let q = self.hidden_size * self.num_heads * hd;
        let kv = self.hidden_size * self.num_kv_heads * hd;
        let o = self.num_heads * hd * self.hidden_size;
        // q/k norms operate over head_dim.
        let qk_norm = 2 * hd;
        let attn = q + 2 * kv + o + qk_norm;
        let mlp = 3 * self.hidden_size * self.intermediate_size;
        let norms = 2 * self.hidden_size;
        attn + mlp + norms
    }
}
