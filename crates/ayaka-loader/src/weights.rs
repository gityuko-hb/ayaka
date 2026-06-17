//! Loaded weights handed to a model factory.

use std::sync::Arc;

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

    /// Split into detached metadata and a shared, owning weight handle.
    /// Convenience for the streaming path (see [`MmapWeights::from_loaded`]).
    pub fn into_shared(self) -> (ModelMetadata, Arc<MmapWeights>) {
        MmapWeights::from_loaded(self)
    }
}

/// Shared, owning handle to a model's weight storage
///
/// Replaces the bare `VarBuilder<'static>` that [`crate::SequentialStreamModel`]
/// previously held by value. The `'static` lifetime on `VarBuilder` denotes
/// *owned* backing storage — candle's `MmapedSafetensors` keeps its mmaps
/// inside an `Arc`, and the GGUF path feeds already-resident tensors via
/// `from_tensors` — **not** a leaked / `Box::leak`-ed buffer. Wrapping that
/// builder in `Arc<MmapWeights>` buys two things:
///
///   * **Deterministic drop** — when the last `Arc` is released the builder is
///     dropped and candle frees its mmaps immediately. No `'static` coercion at
///     call sites, no leak, no reliance on process exit to reclaim mappings.
///   * **Cheap sharing** — the streaming driver and (Phase 2) async prefetch
///     workers clone the `Arc`, never the mmap, so prefetching layer `N+1`
///     while layer `N` computes shares one mapping by reference.
pub struct MmapWeights {
    vb: VarBuilder<'static>,
    weight_bytes: usize,
}

impl MmapWeights {
    /// Wrap a [`LoadedWeights`], returning the detached metadata and a shared
    /// weight handle. Metadata is returned by value because the model factory
    /// consumes it separately from the weight storage.
    pub fn from_loaded(loaded: LoadedWeights) -> (ModelMetadata, Arc<Self>) {
        let LoadedWeights {
            metadata,
            vb,
            weight_bytes,
        } = loaded;
        (metadata, Arc::new(Self { vb, weight_bytes }))
    }

    /// Borrow the `VarBuilder` to pull tensors by name. The borrow is valid for
    /// as long as the `Arc` (and therefore the backing mmap) is held.
    #[inline]
    pub fn var_builder(&self) -> &VarBuilder<'static> {
        &self.vb
    }

    /// Total resident weight bytes (for ledger accounting / diagnostics).
    #[inline]
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    /// Phase 2 shares `Arc<MmapWeights>` across the prefetch worker and the
    /// compute thread, which requires `MmapWeights: Send + Sync`. If this fails
    /// to compile against your candle revision, `VarBuilder<'static>` is not
    /// `Send + Sync` there and the prefetch path must rebuild the `VarBuilder`
    /// per thread instead of sharing one `Arc`.
    #[test]
    fn mmap_weights_is_send_sync() {
        assert_send_sync::<MmapWeights>();
    }
}
