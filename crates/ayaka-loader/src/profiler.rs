//! Dummy-forward activation profiler (design doc §3.6).
//!
//! Runs a forward pass at the target batch/seq and measures the activation
//! memory peak as `free_before - free_after` from the driver, after weights are
//! already resident. Used to size workspace/KV budgets.

use candle_core::{DType, Device, Tensor};

use crate::driver::driver_memory_info;
use crate::error::Result;
use crate::traits::NormalModel;

/// Measured activation footprint of a forward pass, in bytes.
pub fn profile_activation_peak(
    model: &dyn NormalModel,
    device: &Device,
    batch: usize,
    seq_len: usize,
) -> Result<usize> {
    let free_before = driver_memory_info(device)?.free_bytes;

    let input = Tensor::zeros((batch, seq_len), DType::U32, device)?;
    let _logits = model.forward(&input, 0)?;
    device.synchronize()?;

    let free_after = driver_memory_info(device)?.free_bytes;
    Ok(free_before.saturating_sub(free_after))
}
