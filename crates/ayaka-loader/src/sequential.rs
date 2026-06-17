//! Sequential-stream strategy: load one layer, run it, drop it, repeat.
//!
//! This is the design doc's highest-priority path and its #1 risk: each layer's
//! VRAM must actually be reclaimed after `drop`. The model's `Layer` type is
//! responsible for deregistering its bytes from the `MemoryLedger` on `Drop`;
//! this driver enforces the load → forward → drop → synchronize ordering and
//! never holds more than one layer's weights resident at a time.
//!
//! The model no longer holds a bare `VarBuilder<'static>` by value. It holds an
//! [`Arc<MmapWeights>`], which owns the backing mmap and is dropped
//! deterministically when the last clone is released. Cloning the `Arc` (see
//! [`SequentialStreamModel::weights`]) lets a future double-buffering prefetch
//! worker share the same mapping by reference rather than copying it.

use std::sync::Arc;

use candle_core::{Device, Tensor};

use crate::error::LoaderError;
use crate::traits::StreamableModel;
use crate::weights::MmapWeights;

/// Drives a [`StreamableModel`] one layer at a time.
pub struct SequentialStreamModel<M: StreamableModel> {
    model: M,
    weights: Arc<MmapWeights>,
    device: Device,
}

impl<M: StreamableModel> SequentialStreamModel<M> {
    pub fn new(
        model: M,
        weights: Arc<MmapWeights>,
        device: Device,
    ) -> Self {
        Self {
            model,
            weights,
            device,
        }
    }

    /// Forward pass streaming every layer. Holds at most one layer resident.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        // Borrow the shared VarBuilder for the duration of the pass; the mmap
        // stays alive because `self.weights` (the Arc) outlives this borrow.
        let vb = self.weights.var_builder();
        let mut hidden = self.model.embed(input_ids)?;
        for i in 0..self.model.num_layers() {
            let layer = self.model.load_layer(vb, i).map_err(to_candle)?;
            hidden = self
                .model
                .forward_layer(&layer, &hidden, seqlen_offset)?;
            // Drop reclaims this layer's VRAM (Layer::drop deregisters bytes);
            // synchronize so the free completes before the next allocation.
            drop(layer);
            self.device.synchronize()?;
        }
        self.model.norm_and_head(&hidden)
    }

    /// Shared handle to the weight storage. Clone the returned `Arc` to share
    /// the mmap with prefetch workers without copying it.
    pub fn weights(&self) -> Arc<MmapWeights> {
        Arc::clone(&self.weights)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn to_candle(e: LoaderError) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}
