//! Qwen3 quantized model variant.
//!
//! Replaces the 7 float linear projections per decoder layer with
//! [`QuantLinear`] (W4A16): q/k/v/o + gate/up/down. Norms and the shared
//! embed/head stay float since they are small and not worth quantizing. The
//! forward pass is identical to the float [`crate::model::DecoderLayer`] —
//! only the projection matmuls change.

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result as CResult, Tensor};
use candle_nn::{RmsNorm, VarBuilder, rms_norm};

use ayaka_candle::ops::QuantLinear;
use ayaka_loader::{NormalModel, Result as LResult, StreamableModel};
use ayaka_memory::{MemoryPurpose, global_ledger};
use ayaka_quant::{QTensor, WeightRepack};

use crate::config::Qwen3Config;
use crate::model::{Qwen3Shared, apply_rope, repeat_kv};

/// One Qwen3 decoder layer with quantized projections (W4A16).
///
/// Holds [`QuantLinear`] for the 7 projections (q/k/v/o/gate/up/down) and
/// float [`RmsNorm`] for the 4 norm layers. Forward logic is identical to
/// [`crate::model::DecoderLayer`]; only the projection matmuls change.
pub struct QuantDecoderLayer {
    input_ln: RmsNorm,
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_ln: RmsNorm,
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
    weight_bytes: usize,
}

impl Drop for QuantDecoderLayer {
    fn drop(&mut self) {
        let _ = global_ledger().deregister(MemoryPurpose::Weights, self.weight_bytes);
    }
}

impl QuantDecoderLayer {
    /// Load a single quantized layer.
    ///
    /// `vb` should be positioned at `model.layers.{i}` (for the float norms).
    /// `layer_prefix` is the full prefix `"model.layers.{i}"` used to look up
    /// quantized weights in `qtensors`. `device` is where the packed quantized
    /// weights are uploaded.
    pub fn load_quantized(
        vb: &VarBuilder<'static>,
        qtensors: &HashMap<String, QTensor>,
        cfg: &Qwen3Config,
        layer_prefix: &str,
        device: &Device,
    ) -> CResult<Self> {
        let hd = cfg.head_dim;
        let attn_prefix = format!("{layer_prefix}.self_attn");
        let mlp_prefix = format!("{layer_prefix}.mlp");

        let q_proj = build_quant_linear(
            qtensors,
            &format!("{attn_prefix}.q_proj"),
            cfg.hidden_size,
            cfg.num_heads * hd,
            device,
        )?;
        let k_proj = build_quant_linear(
            qtensors,
            &format!("{attn_prefix}.k_proj"),
            cfg.hidden_size,
            cfg.num_kv_heads * hd,
            device,
        )?;
        let v_proj = build_quant_linear(
            qtensors,
            &format!("{attn_prefix}.v_proj"),
            cfg.hidden_size,
            cfg.num_kv_heads * hd,
            device,
        )?;
        let o_proj = build_quant_linear(
            qtensors,
            &format!("{attn_prefix}.o_proj"),
            cfg.num_heads * hd,
            cfg.hidden_size,
            device,
        )?;
        let gate = build_quant_linear(
            qtensors,
            &format!("{mlp_prefix}.gate_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            device,
        )?;
        let up = build_quant_linear(
            qtensors,
            &format!("{mlp_prefix}.up_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            device,
        )?;
        let down = build_quant_linear(
            qtensors,
            &format!("{mlp_prefix}.down_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            device,
        )?;

        let input_ln = rms_norm(cfg.hidden_size, cfg.rms_eps, vb.pp("input_layernorm"))?;
        let q_norm = rms_norm(hd, cfg.rms_eps, vb.pp("self_attn.q_norm"))?;
        let k_norm = rms_norm(hd, cfg.rms_eps, vb.pp("self_attn.k_norm"))?;
        let post_ln = rms_norm(
            cfg.hidden_size,
            cfg.rms_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        // Weight bytes: quantized projection bytes (on-disk packed size) +
        // float norm bytes. The packed size is close to the GPU resident size
        // (U8 quants + F16 scales/mins) and is a good ledger approximation.
        let dtype_bytes = vb.dtype().size_in_bytes();
        let norm_bytes = (2 * cfg.hidden_size + 2 * hd) * dtype_bytes;
        let quant_bytes = [
            format!("{attn_prefix}.q_proj"),
            format!("{attn_prefix}.k_proj"),
            format!("{attn_prefix}.v_proj"),
            format!("{attn_prefix}.o_proj"),
            format!("{mlp_prefix}.gate_proj"),
            format!("{mlp_prefix}.up_proj"),
            format!("{mlp_prefix}.down_proj"),
        ]
        .iter()
        .map(|p| {
            qtensors
                .get(&format!("{p}.weight"))
                .map(|qt| qt.stored_bytes())
                .unwrap_or(0)
        })
        .sum::<usize>();
        let weight_bytes = quant_bytes + norm_bytes;

        let layer = Self {
            input_ln,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            post_ln,
            gate,
            up,
            down,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: hd,
            n_rep: cfg.n_rep(),
            weight_bytes,
        };
        global_ledger().register(MemoryPurpose::Weights, layer.weight_bytes);
        Ok(layer)
    }

    /// Forward pass — identical logic to `DecoderLayer::forward` but with
    /// [`QuantLinear`] projections.
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

/// Build a [`QuantLinear`] from a quantized weight in `qtensors`.
///
/// `prefix` is the projection path (e.g. `"model.layers.0.self_attn.q_proj"`);
/// the lookup key is `"{prefix}.weight"`. Validates that the repacked
/// dimensions match `in_features` and `out_features`.
fn build_quant_linear(
    qtensors: &HashMap<String, QTensor>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device,
) -> CResult<QuantLinear> {
    let key = format!("{prefix}.weight");
    let qt = qtensors
        .get(&key)
        .ok_or_else(|| candle_core::Error::Msg(format!("quantized weight not found: {key}")))?;
    let rw = qt
        .repack()
        .map_err(|e| candle_core::Error::Msg(format!("repack failed for {key}: {e}")))?;
    if rw.in_features != in_features || rw.out_features != out_features {
        candle_core::bail!(
            "quantized weight {key}: expected in={in_features} out={out_features} \
             but got in={} out={}",
            rw.in_features,
            rw.out_features
        );
    }
    QuantLinear::from_repacked(&rw, None, device)
}

/// All-resident quantized Qwen3 model (FullGpu / PartialOffload consumer).
pub struct Qwen3QuantModel {
    shared: Qwen3Shared,
    layers: Vec<QuantDecoderLayer>,
}

impl Qwen3QuantModel {
    /// Load every quantized layer plus the shared parts.
    pub fn load_quantized(
        vb: &VarBuilder<'static>,
        qtensors: &HashMap<String, QTensor>,
        cfg: Qwen3Config,
        dtype: DType,
        device: &Device,
    ) -> LResult<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(QuantDecoderLayer::load_quantized(
                &vb.pp(&prefix),
                qtensors,
                &cfg,
                &prefix,
                device,
            )?);
        }
        let shared = Qwen3Shared::load(vb, cfg, dtype, device)?;
        Ok(Self { shared, layers })
    }
}

impl NormalModel for Qwen3QuantModel {
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

/// Streaming quantized Qwen3 model: holds the shared parts plus the packed
/// quantized weights, loads each decoder layer on demand. Drives the
/// sequential-stream strategy for quantized models.
pub struct Qwen3QuantStreamer {
    shared: Qwen3Shared,
    qtensors: HashMap<String, QTensor>,
}

impl Qwen3QuantStreamer {
    /// Load shared parts and take ownership of the quantized weight map.
    pub fn load_quantized(
        vb: &VarBuilder<'static>,
        qtensors: HashMap<String, QTensor>,
        cfg: Qwen3Config,
        dtype: DType,
        device: &Device,
    ) -> LResult<Self> {
        let shared = Qwen3Shared::load(vb, cfg, dtype, device)?;
        Ok(Self { shared, qtensors })
    }
}

impl StreamableModel for Qwen3QuantStreamer {
    type Layer = QuantDecoderLayer;

    fn num_layers(&self) -> usize {
        self.shared.cfg.num_layers
    }

    fn embed(
        &self,
        input_ids: &Tensor,
    ) -> CResult<Tensor> {
        self.shared.embed(input_ids)
    }

    /// Build a quantized layer from the streamer's stored `qtensors` and the
    /// shared `vb`. The float norms come from `vb`; the 7 projections come
    /// from the stored packed quantized weights.
    fn load_layer(
        &self,
        vb: &VarBuilder<'static>,
        idx: usize,
    ) -> LResult<Self::Layer> {
        let prefix = format!("model.layers.{idx}");
        Ok(QuantDecoderLayer::load_quantized(
            &vb.pp(&prefix),
            &self.qtensors,
            &self.shared.cfg,
            &prefix,
            &self.shared.device,
        )?)
    }

    fn load_layer_quantized(
        &self,
        vb: &VarBuilder<'static>,
        qtensors: &HashMap<String, QTensor>,
        idx: usize,
    ) -> LResult<Self::Layer> {
        let _ = qtensors;
        self.load_layer(vb, idx)
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
