//! Sequential-stream strategy: load one layer, run it, drop it, repeat.
//!
//! This is the design doc's highest-priority path and its #1 risk: each layer's
//! VRAM must actually be reclaimed after `drop`. The model's `Layer` type is
//! responsible for deregistering its bytes from the `MemoryLedger` on `Drop`;
//! this driver enforces the load → forward → drop → synchronize ordering and
//! never holds more than one layer's weights resident at a time.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::LoaderError;
use crate::traits::StreamableModel;

/// Drives a [`StreamableModel`] one layer at a time.
pub struct SequentialStreamModel<M: StreamableModel> {
    model: M,
    vb: VarBuilder<'static>,
    device: Device,
}

impl<M: StreamableModel> SequentialStreamModel<M> {
    pub fn new(
        model: M,
        vb: VarBuilder<'static>,
        device: Device,
    ) -> Self {
        Self { model, vb, device }
    }

    /// Forward pass streaming every layer. Holds at most one layer resident.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        let mut hidden = self.model.embed(input_ids)?;
        for i in 0..self.model.num_layers() {
            let layer = self
                .model
                .load_layer(&self.vb, i)
                .map_err(to_candle)?;
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

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn to_candle(e: LoaderError) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}
