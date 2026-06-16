//! Model-agnostic traits. The loader depends only on these; concrete models
//! (in `ayaka-model`) implement them, keeping the dependency graph acyclic.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::device_map::DeviceMap;
use crate::error::Result;
use crate::weights::LoadedWeights;

/// Options controlling how a model is loaded.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Compute dtype that quantized/other weights are materialized to.
    pub dtype: DType,
    /// Target compute device.
    pub device: Device,
    /// Fraction of total VRAM the loader may use (headroom = 1 - this).
    pub gpu_memory_utilization: f64,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            dtype: DType::F16,
            device: Device::Cpu,
            gpu_memory_utilization: 0.9,
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
}
