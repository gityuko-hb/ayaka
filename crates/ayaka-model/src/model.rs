//! Qwen3 reference model.
//!
//! Exercises the load / memory / streaming mechanics, not decode throughput.
//! Attention uses vanilla Candle (matmul + softmax) so the model is testable on
//! CPU today; the `cuda` feature is where ayaka's fused kernels would slot in.
//! Numeric parity with HF/Candle is intentionally out of scope here.

use candle_core::{D, DType, Device, IndexOp, Module, Result as CResult, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm};

use ayaka_loader::{NormalModel, Result as LResult, StreamableModel};
use ayaka_memory::{MemoryPurpose, global_ledger};

use crate::config::Qwen3Config;

/// Components shared by both the resident and streaming models.
pub struct Qwen3Shared {
    embed: Embedding,
    norm: RmsNorm,
    lm_head: Linear,
    /// RoPE caches, shape `[max_pos, head_dim]`.
    cos: Tensor,
    sin: Tensor,
    pub(crate) cfg: Qwen3Config,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
}

impl Qwen3Shared {
    /// Load embeddings, final norm, head, and build the RoPE cache.
    pub fn load(
        vb: &VarBuilder<'static>,
        cfg: Qwen3Config,
        dtype: DType,
        device: &Device,
    ) -> CResult<Self> {
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_eps, vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let (cos, sin) = rope_cache(&cfg, dtype, device)?;
        Ok(Self {
            embed,
            norm,
            lm_head,
            cos,
            sin,
            cfg,
            device: device.clone(),
            dtype,
        })
    }

    pub(crate) fn embed(
        &self,
        input_ids: &Tensor,
    ) -> CResult<Tensor> {
        self.embed.forward(input_ids)
    }

    pub(crate) fn norm_and_head(
        &self,
        hidden: &Tensor,
    ) -> CResult<Tensor> {
        self.lm_head.forward(&self.norm.forward(hidden)?)
    }

    /// `(cos, sin)` slices for positions `offset..offset+seq`, shaped `[seq, head_dim]`.
    pub(crate) fn rope_slice(
        &self,
        offset: usize,
        seq: usize,
    ) -> CResult<(Tensor, Tensor)> {
        let cos = self.cos.i((offset..offset + seq, ..))?;
        let sin = self.sin.i((offset..offset + seq, ..))?;
        Ok((cos, sin))
    }

    /// Causal mask `[seq, seq]` (0 on/below diagonal, -inf above) in model dtype.
    pub(crate) fn causal_mask(
        &self,
        seq: usize,
    ) -> CResult<Tensor> {
        let mut data = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(data, (seq, seq), &self.device)?.to_dtype(self.dtype)
    }
}

/// Build RoPE `cos`/`sin` caches of shape `[max_pos, head_dim]`.
fn rope_cache(
    cfg: &Qwen3Config,
    dtype: DType,
    device: &Device,
) -> CResult<(Tensor, Tensor)> {
    let half = cfg.head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| {
            1f32 / cfg
                .rope_theta
                .powf((2 * j) as f32 / cfg.head_dim as f32)
        })
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_position_embeddings, 1))?;
    let freqs = t.matmul(&inv_freq)?; // [max_pos, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [max_pos, head_dim]
    Ok((emb.cos()?.to_dtype(dtype)?, emb.sin()?.to_dtype(dtype)?))
}

/// `cat(-x2, x1)` over the last dim — the neox "rotate half".
fn rotate_half(x: &Tensor) -> CResult<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply RoPE to `x` of shape `[b, heads, seq, head_dim]` given `cos`/`sin`
/// of shape `[seq, head_dim]`.
pub(crate) fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> CResult<Tensor> {
    let (_b, _h, s, d) = x.dims4()?;
    let cos = cos.reshape((1, 1, s, d))?;
    let sin = sin.reshape((1, 1, s, d))?;
    x.broadcast_mul(&cos)?
        .add(&rotate_half(x)?.broadcast_mul(&sin)?)
}

/// One Qwen3 decoder layer. Its weight bytes are registered in the global
/// [`MemoryLedger`] on load and deregistered on `Drop`, so the sequential
/// strategy can assert that streaming returns the ledger to baseline.
pub struct DecoderLayer {
    input_ln: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_ln: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
    weight_bytes: usize,
}

impl Drop for DecoderLayer {
    fn drop(&mut self) {
        let _ = global_ledger().deregister(MemoryPurpose::Weights, self.weight_bytes);
    }
}

impl DecoderLayer {
    /// Load a single layer from `vb` positioned at `model.layers.{i}`.
    pub fn load(
        vb: &VarBuilder<'static>,
        cfg: &Qwen3Config,
        dtype_bytes: usize,
    ) -> CResult<Self> {
        let hd = cfg.head_dim;
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        let layer = Self {
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_eps, vb.pp("input_layernorm"))?,
            q_proj: linear_no_bias(cfg.hidden_size, cfg.num_heads * hd, attn.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, cfg.num_kv_heads * hd, attn.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, cfg.num_kv_heads * hd, attn.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.num_heads * hd, cfg.hidden_size, attn.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_eps, attn.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_eps, attn.pp("k_norm"))?,
            post_ln: rms_norm(
                cfg.hidden_size,
                cfg.rms_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, mlp.pp("down_proj"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: hd,
            n_rep: cfg.n_rep(),
            weight_bytes: cfg.layer_param_count() * dtype_bytes,
        };
        global_ledger().register(MemoryPurpose::Weights, layer.weight_bytes);
        Ok(layer)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CResult<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let residual = hidden;
        let x = self.input_ln.forward(hidden)?;

        // Project and split into heads: [b, heads, s, head_dim].
        let q = self
            .q_proj
            .forward(&x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(&x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(&x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Qwen3 per-head RMSNorm on q and k, then RoPE.
        let q = self
            .q_norm
            .forward(&q)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_norm
            .forward(&k)?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat kv heads up to query heads.
        let k = repeat_kv(&k, self.n_rep)?;
        let v = repeat_kv(&v, self.n_rep)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [b, heads, s, head_dim]

        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        let attn_out = self.o_proj.forward(&ctx)?;
        let hidden = (residual + attn_out)?;

        // SwiGLU MLP.
        let residual = &hidden;
        let x = self.post_ln.forward(&hidden)?;
        let swiglu = (candle_nn::ops::silu(&self.gate.forward(&x)?)? * self.up.forward(&x)?)?;
        let mlp_out = self.down.forward(&swiglu)?;
        residual + mlp_out
    }
}

/// Expand `[b, kv_heads, s, d]` to `[b, kv_heads * n_rep, s, d]`.
pub(crate) fn repeat_kv(
    x: &Tensor,
    n_rep: usize,
) -> CResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv, n_rep, s, d))?
        .reshape((b, kv * n_rep, s, d))
}

/// All-resident Qwen3 model (FullGpu / PartialOffload consumer).
pub struct Qwen3Model {
    shared: Qwen3Shared,
    layers: Vec<DecoderLayer>,
}

impl Qwen3Model {
    /// Load every layer plus the shared parts.
    pub fn load(
        vb: &VarBuilder<'static>,
        cfg: Qwen3Config,
        dtype: DType,
        device: &Device,
    ) -> LResult<Self> {
        let dtype_bytes = dtype.size_in_bytes();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(
                &vb.pp(format!("model.layers.{i}")),
                &cfg,
                dtype_bytes,
            )?);
        }
        let shared = Qwen3Shared::load(vb, cfg, dtype, device)?;
        Ok(Self { shared, layers })
    }
}

impl NormalModel for Qwen3Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> CResult<Tensor> {
        let mut hidden = self.shared.embed(input_ids)?;
        let s = hidden.dim(1)?;
        let (cos, sin) = self.shared.rope_slice(seqlen_offset, s)?;
        let mask = self.shared.causal_mask(s)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        self.shared.norm_and_head(&hidden)
    }

    fn device(&self) -> &Device {
        &self.shared.device
    }
}

/// Streaming Qwen3 model: holds only the shared parts, loads each decoder layer
/// on demand. Drives the sequential-stream strategy.
pub struct Qwen3Streamer {
    shared: Qwen3Shared,
    dtype_bytes: usize,
}

impl Qwen3Streamer {
    pub fn load(
        vb: &VarBuilder<'static>,
        cfg: Qwen3Config,
        dtype: DType,
        device: &Device,
    ) -> LResult<Self> {
        let dtype_bytes = dtype.size_in_bytes();
        let shared = Qwen3Shared::load(vb, cfg, dtype, device)?;
        Ok(Self {
            shared,
            dtype_bytes,
        })
    }
}

impl StreamableModel for Qwen3Streamer {
    type Layer = DecoderLayer;

    fn num_layers(&self) -> usize {
        self.shared.cfg.num_layers
    }

    fn embed(
        &self,
        input_ids: &Tensor,
    ) -> CResult<Tensor> {
        self.shared.embed(input_ids)
    }

    fn load_layer(
        &self,
        vb: &VarBuilder<'static>,
        idx: usize,
    ) -> LResult<Self::Layer> {
        Ok(DecoderLayer::load(
            &vb.pp(format!("model.layers.{idx}")),
            &self.shared.cfg,
            self.dtype_bytes,
        )?)
    }

    fn forward_layer(
        &self,
        layer: &Self::Layer,
        hidden: &Tensor,
        seqlen_offset: usize,
    ) -> CResult<Tensor> {
        let s = hidden.dim(1)?;
        let (cos, sin) = self.shared.rope_slice(seqlen_offset, s)?;
        let mask = self.shared.causal_mask(s)?;
        layer.forward(hidden, &cos, &sin, &mask)
    }

    fn norm_and_head(
        &self,
        hidden: &Tensor,
    ) -> CResult<Tensor> {
        self.shared.norm_and_head(hidden)
    }
}
