//! Model-agnostic traits. The loader depends only on these; concrete models
//! (in `ayaka-model`) implement them, keeping the dependency graph acyclic.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use ayaka_quant::QTensor;

use crate::device_map::DeviceMap;
use crate::error::Result;
use crate::weights::{LoadedQuantWeights, LoadedWeights};

/// Options controlling how a model is loaded.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Compute dtype that quantized/other weights are materialized to.
    pub dtype: DType,
    /// Target compute device.
    pub device: Device,
    /// Fraction of total VRAM the loader may use (headroom = 1 - this).
    pub gpu_memory_utilization: f64,
    /// Max tokens per batch iteration — drives the activation memory reserve
    /// in [`crate::strategy::gpu::partial_offload`]. Set from `EnvConfig` by the
    /// caller. Default: 2048.
    pub max_num_batched_tokens: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            dtype: DType::F16,
            device: Device::Cpu,
            gpu_memory_utilization: 0.9,
            max_num_batched_tokens: 2048,
        }
    }
}

/// A loaded, runnable model. Object-safe so the registry can hand back
/// `Box<dyn NormalModel>`.
pub trait NormalModel {
    /// Run a forward pass over `input_ids` (shape `[batch, seq]`), returning
    /// logits. `seqlen_offset` is the position of the first token (0 for prefill).
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor>;

    /// The device the model computes on.
    fn device(&self) -> &Device;
}

/// A model that can load and run one decoder layer at a time, enabling the
/// sequential-stream strategy. The associated `Layer` type holds a single
/// layer's resident weights so it can be dropped to reclaim VRAM.
pub trait StreamableModel {
    /// Per-layer weight bundle (dropping it frees that layer's VRAM).
    type Layer;

    fn num_layers(&self) -> usize;

    /// Embed input token ids into hidden states.
    fn embed(
        &self,
        input_ids: &Tensor,
    ) -> candle_core::Result<Tensor>;

    /// Materialize layer `idx`'s weights from `vb` onto the compute device.
    fn load_layer(
        &self,
        vb: &VarBuilder<'static>,
        idx: usize,
    ) -> Result<Self::Layer>;

    /// Materialize layer `idx`'s weights from `vb` + optional quantized
    /// tensors. Default implementation ignores `qtensors` and delegates to
    /// [`load_layer`](Self::load_layer). Models that support quantized
    /// streaming override this to build `QuantLinear` layers from the
    /// `QTensor` entries.
    fn load_layer_quantized(
        &self,
        vb: &VarBuilder<'static>,
        qtensors: &HashMap<String, QTensor>,
        idx: usize,
    ) -> Result<Self::Layer> {
        let _ = qtensors; // unused in default impl
        self.load_layer(vb, idx)
    }

    /// Run hidden states through a single loaded layer.
    fn forward_layer(
        &self,
        layer: &Self::Layer,
        hidden: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor>;

    /// Apply the final norm and language-model head to produce logits.
    fn norm_and_head(
        &self,
        hidden: &Tensor,
    ) -> candle_core::Result<Tensor>;
}

/// Builds a concrete model from loaded weights. Registered in `ayaka-registry`
/// keyed by `arch_id`.
pub trait ModelLoaderFactory: Send + Sync {
    /// Canonical architecture id this factory handles (e.g. `"qwen3"`).
    fn arch_id(&self) -> &'static str;

    /// Construct a runnable model from loaded weights and a placement map.
    fn load(
        &self,
        weights: LoadedWeights,
        device_map: &DeviceMap,
    ) -> Result<Box<dyn NormalModel>>;

    /// Construct a runnable model from quantized weights (packed `QTensor` +
    /// float `VarBuilder`). Default implementation returns an error; factories
    /// that support quantized loading override this.
    fn load_quantized(
        &self,
        _weights: LoadedQuantWeights,
        _device_map: &DeviceMap,
    ) -> Result<Box<dyn NormalModel>> {
        Err(crate::error::LoaderError::UnsupportedArch(format!(
            "architecture {} does not support quantized loading",
            self.arch_id()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ModelMetadata;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    struct DummyFactory;

    impl ModelLoaderFactory for DummyFactory {
        fn arch_id(&self) -> &'static str {
            "dummy"
        }
        fn load(
            &self,
            _weights: LoadedWeights,
            _device_map: &DeviceMap,
        ) -> Result<Box<dyn NormalModel>> {
            unreachable!()
        }
    }

    #[test]
    fn default_load_quantized_returns_unsupported() {
        let factory = DummyFactory;
        let meta = ModelMetadata {
            arch_id: "dummy".into(),
            hidden_size: 1,
            intermediate_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: None,
            head_dim: None,
            vocab_size: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 1,
            tie_word_embeddings: false,
            mla: None,
            moe: None,
            quant_config: None,
        };
        let vb = VarBuilder::from_tensors(HashMap::new(), candle_core::DType::F32, &Device::Cpu);
        let weights = LoadedQuantWeights::new(meta, vb, HashMap::new(), HashMap::new(), 0);
        let device_map = DeviceMap::all_gpu(Device::Cpu, 1);
        let result = factory.load_quantized(weights, &device_map);
        match result {
            Err(crate::error::LoaderError::UnsupportedArch(msg)) => {
                assert!(msg.contains("dummy"), "msg should mention arch id: {msg}");
            },
            Err(other) => panic!("expected UnsupportedArch, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
