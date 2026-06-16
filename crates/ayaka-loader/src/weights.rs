//! Loaded weights handed to a model factory.

use candle_nn::VarBuilder;

use crate::metadata::ModelMetadata;

/// Parsed metadata plus a [`VarBuilder`] rooted at the model's tensor namespace.
///
/// Both the safetensors and GGUF paths converge on this type: safetensors via
/// `VarBuilder::from_mmaped_safetensors`, GGUF via dequantized tensors fed
/// through `VarBuilder::from_tensors`. Models pull weights by name from `vb`.
pub struct LoadedWeights {
    pub metadata: ModelMetadata,
    pub vb: VarBuilder<'static>,
    /// Total resident weight bytes, for `MemoryLedger` accounting.
    pub weight_bytes: usize,
}

impl LoadedWeights {
    pub fn new(
        metadata: ModelMetadata,
        vb: VarBuilder<'static>,
        weight_bytes: usize,
    ) -> Self {
        Self {
            metadata,
            vb,
            weight_bytes,
        }
    }
}
