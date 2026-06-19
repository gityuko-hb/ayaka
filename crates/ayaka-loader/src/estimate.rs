//! Memory estimation (design doc REQ1): weight footprint, KV-cache cost, and
//! auto-fit of how many layers fit on the GPU.

use ayaka_quant::QuantScheme;

use crate::metadata::ModelMetadata;

/// Estimated memory breakdown for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// Embeddings + lm_head + final norm, in bytes.
    pub fixed_bytes: usize,
    /// One decoder layer's weights, in bytes (dense path; MoE layers differ).
    pub per_layer_bytes: usize,
    /// All weights, in bytes (fixed + every layer, dense/MoE as configured).
    pub total_weight_bytes: usize,
    /// KV-cache bytes for a single token across all layers.
    pub kv_bytes_per_token: usize,
}

/// Computes [`MemoryEstimate`] from metadata, weight scheme, and KV dtype size.
pub struct MemoryEstimator<'a> {
    metadata: &'a ModelMetadata,
    weight_scheme: QuantScheme,
    kv_dtype_bytes: usize,
    /// Fraction of total VRAM usable (headroom = 1 - this).
    gpu_memory_utilization: f64,
}

impl<'a> MemoryEstimator<'a> {
    pub fn new(
        metadata: &'a ModelMetadata,
        weight_scheme: QuantScheme,
        kv_dtype_bytes: usize,
        gpu_memory_utilization: f64,
    ) -> Self {
        Self {
            metadata,
            weight_scheme,
            kv_dtype_bytes,
            gpu_memory_utilization,
        }
    }

    fn bytes(
        &self,
        params: usize,
    ) -> usize {
        (params as f64 * self.weight_scheme.bytes_per_weight()).ceil() as usize
    }

    /// Parameters in embeddings + lm_head + final norm.
    fn fixed_params(&self) -> usize {
        let m = self.metadata;
        let embed = m.vocab_size * m.hidden_size;
        let head = if m.tie_word_embeddings {
            0
        } else {
            embed
        };
        embed + head + m.hidden_size
    }

    /// Parameters in one dense decoder layer (GQA or MHA).
    fn dense_layer_params(&self) -> usize {
        let m = self.metadata;
        let hd = m.head_dim();
        let q = m.hidden_size * m.num_attention_heads * hd;
        let kv = m.hidden_size * m.num_kv_heads() * hd; // k and v each
        let o = m.num_attention_heads * hd * m.hidden_size;
        let attn = q + 2 * kv + o;
        // SwiGLU MLP: gate + up + down.
        let mlp = 3 * m.hidden_size * m.intermediate_size;
        let norms = 2 * m.hidden_size;
        attn + mlp + norms
    }

    /// Parameters in one MoE decoder layer (router + experts + shared experts).
    fn moe_layer_params(&self) -> usize {
        let m = self.metadata;
        let Some(moe) = &m.moe else {
            return self.dense_layer_params();
        };
        let hd = m.head_dim();
        let q = m.hidden_size * m.num_attention_heads * hd;
        let kv = m.hidden_size * m.num_kv_heads() * hd;
        let o = m.num_attention_heads * hd * m.hidden_size;
        let attn = q + 2 * kv + o;

        let per_expert = 3 * m.hidden_size * moe.moe_intermediate_size;
        let experts = moe.n_routed_experts * per_expert;
        let shared = moe.n_shared_experts * per_expert;
        let router = m.hidden_size * moe.n_routed_experts;
        let norms = 2 * m.hidden_size;
        attn + experts + shared + router + norms
    }

    /// KV-cache bytes for one token across all layers.
    ///
    /// GQA: `2 * layers * num_kv_heads * head_dim * dtype`.
    /// MLA: `layers * (kv_lora_rank + qk_rope_head_dim) * dtype` (compressed
    /// latent shared across heads — no `× num_kv_heads`).
    pub fn kv_bytes_per_token(&self) -> usize {
        let m = self.metadata;
        let per_layer = match &m.mla {
            Some(mla) => mla.kv_lora_rank + mla.qk_rope_head_dim,
            None => 2 * m.num_kv_heads() * m.head_dim(),
        };
        per_layer * m.num_hidden_layers * self.kv_dtype_bytes
    }

    /// Full estimate.
    pub fn estimate(&self) -> MemoryEstimate {
        let m = self.metadata;
        let fixed_bytes = self.bytes(self.fixed_params());
        let per_layer_bytes = self.bytes(self.dense_layer_params());

        let first_dense = m
            .moe
            .as_ref()
            .map(|moe| moe.first_k_dense_replace)
            .unwrap_or(m.num_hidden_layers);
        let mut layers_bytes = 0usize;
        for i in 0..m.num_hidden_layers {
            let params = if i < first_dense {
                self.dense_layer_params()
            } else {
                self.moe_layer_params()
            };
            layers_bytes += self.bytes(params);
        }

        MemoryEstimate {
            fixed_bytes,
            per_layer_bytes,
            total_weight_bytes: fixed_bytes + layers_bytes,
            kv_bytes_per_token: self.kv_bytes_per_token(),
        }
    }

    /// Auto-fit how many decoder layers fit on the GPU (design doc §1.6),
    /// given driver `free`/`total` bytes and bytes reserved for KV + activations.
    ///
    /// Usable budget = `free - (1 - util) * total`, minus the fixed weights and
    /// the caller's reservation; the remainder divided by per-layer bytes.
    pub fn fit_gpu_layers(
        &self,
        free_bytes: usize,
        total_bytes: usize,
        reserve_bytes: usize,
    ) -> usize {
        let est = self.estimate();
        let headroom = ((1.0 - self.gpu_memory_utilization) * total_bytes as f64) as usize;
        let usable = free_bytes
            .saturating_sub(headroom)
            .saturating_sub(est.fixed_bytes)
            .saturating_sub(reserve_bytes);
        if est.per_layer_bytes == 0 {
            return 0;
        }
        (usable / est.per_layer_bytes).min(self.metadata.num_hidden_layers)
    }

    /// vLLM formula: maximum number of KV cache blocks that fit in the
    /// remaining VRAM after weights and activations.
    ///
    /// `num_gpu_blocks = (usable - weight_bytes - activation_peak) / block_bytes`
    ///
    /// Where `block_bytes = kv_bytes_per_token * block_size` (2 K/V tensors
    /// per layer × layers × per-token-per-layer × block_size).
    pub fn num_gpu_blocks(
        &self,
        usable_bytes: usize,
        weight_bytes: usize,
        activation_peak_bytes: usize,
        block_bytes: usize,
    ) -> usize {
        if block_bytes == 0 {
            return 0;
        }
        usable_bytes
            .saturating_sub(weight_bytes)
            .saturating_sub(activation_peak_bytes)
            / block_bytes
    }

    /// Bytes occupied by one KV cache block (all layers, both K and V).
    ///
    /// GQA: `2 * layers * num_kv_heads * head_dim * dtype * block_size`
    /// MLA: `(kv_lora_rank + qk_rope_head_dim) * layers * dtype * block_size`
    pub fn kv_block_bytes(
        &self,
        block_size: usize,
    ) -> usize {
        self.kv_bytes_per_token() * block_size
    }
}

/// End-to-end computation of the maximum number of KV cache blocks that fit
/// in GPU memory, using the vLLM formula with Phase 1.2 activation reserve.
///
/// This is the convenience wrapper that ties together:
/// - [`MemoryEstimator::kv_block_bytes`] (block size -> bytes, MLA-aware)
/// - [`MemoryEstimator::num_gpu_blocks`] (vLLM formula)
/// - The static activation reserve from `max_num_batched_tokens`
///
/// Returns `(num_blocks, activation_reserve_bytes, block_bytes)` so callers
/// can log the breakdown.
#[allow(clippy::too_many_arguments)]
pub fn compute_num_gpu_blocks(
    metadata: &ModelMetadata,
    weight_bytes: usize,
    free_bytes: usize,
    total_bytes: usize,
    gpu_memory_utilization: f64,
    max_num_batched_tokens: usize,
    kv_dtype_bytes: usize,
    block_size: usize,
) -> (usize, usize, usize) {
    let est = MemoryEstimator::new(
        metadata,
        QuantScheme::F16,
        kv_dtype_bytes,
        gpu_memory_utilization,
    );

    // Static activation reserve: max_tokens x hidden x dtype x 2
    let activation_reserve = max_num_batched_tokens * metadata.hidden_size * kv_dtype_bytes * 2;

    let block_bytes = est.kv_block_bytes(block_size);

    // Usable = free - headroom (1 - util) x total
    let headroom = ((1.0 - gpu_memory_utilization) * total_bytes as f64) as usize;
    let usable = free_bytes.saturating_sub(headroom);

    let num_blocks = est.num_gpu_blocks(usable, weight_bytes, activation_reserve, block_bytes);

    (num_blocks, activation_reserve, block_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{MlaConfig, ModelMetadata};

    fn qwen3_0_6b() -> ModelMetadata {
        ModelMetadata {
            arch_id: "qwen3".into(),
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 40960,
            tie_word_embeddings: true,
            mla: None,
            moe: None,
            quant_config: None,
        }
    }

    #[test]
    fn dense_layer_and_total_params_match_hand_calc() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        let e = est.estimate();
        // Hand-computed: per-layer = 15,730,688 params -> *2 bytes (F16).
        assert_eq!(e.per_layer_bytes, 15_730_688 * 2);
        // fixed (tied head) = embed + final norm = 151936*1024 + 1024.
        assert_eq!(e.fixed_bytes, (151_936 * 1024 + 1024) * 2);
        // total params ≈ 0.596B.
        let total_params = e.total_weight_bytes / 2;
        assert_eq!(total_params, 596_042_752);
    }

    #[test]
    fn kv_per_token_gqa() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        // 2 * 28 layers * 8 kv_heads * 128 head_dim * 2 bytes.
        assert_eq!(est.kv_bytes_per_token(), 2 * 28 * 8 * 128 * 2);
    }

    #[test]
    fn kv_per_token_mla_has_no_kv_head_factor() {
        let mut m = qwen3_0_6b();
        m.mla = Some(MlaConfig {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            q_lora_rank: Some(1536),
        });
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        // (512 + 64) * 28 layers * 2 bytes — no num_kv_heads multiplier.
        assert_eq!(est.kv_bytes_per_token(), (512 + 64) * 28 * 2);
    }

    #[test]
    fn fit_gpu_layers_respects_headroom_and_caps() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        let e = est.estimate();
        // total = 8 GiB, util 0.9 -> headroom 0.8 GiB. free = 8 GiB.
        let total = 8 * 1024 * 1024 * 1024;
        let free = total;
        let n = est.fit_gpu_layers(free, total, 0);
        // Enough room for all 28 layers -> capped at 28.
        assert_eq!(n, 28);

        // Room for fixed + exactly 3.5 layers (after headroom) -> floor to 3.
        let headroom = ((1.0 - 0.9) * total as f64) as usize;
        let small_free = headroom + e.fixed_bytes + e.per_layer_bytes * 3 + e.per_layer_bytes / 2;
        let n2 = est.fit_gpu_layers(small_free, total, 0);
        assert_eq!(n2, 3);
    }

    #[test]
    fn num_gpu_blocks_vllm_formula() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        // usable=80GB, weights=14GB, activation=2GB, block_bytes=8MB
        // -> (80-14-2)GB / 8MB = 64GB / 8MB = 8192 blocks
        let usable = 80 * 1024 * 1024 * 1024;
        let weights = 14 * 1024 * 1024 * 1024;
        let activation = 2 * 1024 * 1024 * 1024;
        let block_bytes = 8 * 1024 * 1024;
        assert_eq!(
            est.num_gpu_blocks(usable, weights, activation, block_bytes),
            8192
        );
    }

    #[test]
    fn num_gpu_blocks_zero_block_bytes_returns_zero() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        assert_eq!(est.num_gpu_blocks(1_000_000, 0, 0, 0), 0);
    }

    #[test]
    fn num_gpu_blocks_saturates_when_insufficient() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        // weights + activation > usable -> 0 blocks
        assert_eq!(est.num_gpu_blocks(1000, 2000, 2000, 16), 0);
    }

    #[test]
    fn kv_block_bytes_matches_per_token_formula() {
        let m = qwen3_0_6b();
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        let per_token = est.kv_bytes_per_token();
        // block_size = 16 (default)
        assert_eq!(est.kv_block_bytes(16), per_token * 16);
    }

    #[test]
    fn kv_block_bytes_mla_uses_compressed_latent() {
        let mut m = qwen3_0_6b();
        m.mla = Some(MlaConfig {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            q_lora_rank: Some(1536),
        });
        let est = MemoryEstimator::new(&m, QuantScheme::F16, 2, 0.9);
        // MLA: (512+64) * 28 * 2 bytes * 16 tokens
        assert_eq!(est.kv_block_bytes(16), (512 + 64) * 28 * 2 * 16);
    }

    #[test]
    fn compute_num_gpu_blocks_end_to_end() {
        let m = qwen3_0_6b();
        // usable=72GB (80GB free - 8GB headroom), weights=14GB,
        // activation~8MB, block_bytes=1.75MB
        // -> (72-14-0.008)GB / 1.75MB ~= 33933 blocks
        let (blocks, reserve, block_bytes) = compute_num_gpu_blocks(
            &m,
            14 * 1024 * 1024 * 1024, // 14 GB weights
            80 * 1024 * 1024 * 1024, // 80 GB free
            80 * 1024 * 1024 * 1024, // 80 GB total
            0.9,                     // utilization
            2048,                    // max_tokens
            2,                       // F16 dtype bytes
            16,                      // block_size
        );
        assert!(blocks > 0);
        assert_eq!(reserve, 2048 * 1024 * 2 * 2); // 8 MB
        // kv_block_bytes = 2 * 28 * 8 * 128 * 2 * 16 = 1,835,008
        assert_eq!(block_bytes, 2 * 28 * 8 * 128 * 2 * 16);
    }
}
