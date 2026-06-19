//! Dummy-forward activation profiler
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

/// Profile activation peak at the worst-case prefill point.
///
/// Production schedulers cap the total tokens per forward iteration via
/// `max_num_batched_tokens`. The worst-case activation footprint occurs at
/// this saturation point: batch = `max_batch_size`, seq_len = `max_tokens / max_batch`.
///
/// Decode phase (1 token per request) always has smaller activations than
/// prefill, so we only profile prefill.
///
/// Returns the measured activation peak in bytes, or 0 if profiling fails
/// (callers fall back to `reserve = 0`).
pub fn profile_prefill_peak(
    model: &dyn NormalModel,
    device: &Device,
    max_num_batched_tokens: usize,
    max_batch_size: usize,
) -> Result<usize> {
    let batch = max_batch_size.max(1);
    let seq_len = (max_num_batched_tokens / batch).max(1);
    profile_activation_peak(model, device, batch, seq_len)
}
