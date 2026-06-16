//! Load-strategy selection and (GPU-only) execution.
//!
//! The selection math and format dispatch are pure and CPU-testable; the
//! strategies that actually move weights to VRAM are gated behind `cuda`.

use std::path::Path;

use candle_core::{DType, Device};

use crate::error::{LoaderError, Result};
use crate::weights::LoadedWeights;

/// Which hardware strategy to use for a load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyKind {
    /// All weights resident in VRAM.
    FullGpu,
    /// Some layers in VRAM, the rest staged from host RAM.
    PartialOffload,
    /// Layers streamed one at a time from disk.
    SequentialStream,
}

/// Choose a strategy from the decision matrix (design doc §2): compare the
/// model's weight footprint against usable VRAM (`util × total`) and then
/// against usable VRAM + host RAM.
pub fn select_strategy(
    model_weight_bytes: usize,
    total_vram_bytes: usize,
    host_ram_bytes: usize,
    gpu_memory_utilization: f64,
) -> StrategyKind {
    let usable_vram = (total_vram_bytes as f64 * gpu_memory_utilization) as usize;
    if model_weight_bytes <= usable_vram {
        StrategyKind::FullGpu
    } else if model_weight_bytes <= usable_vram + host_ram_bytes {
        StrategyKind::PartialOffload
    } else {
        StrategyKind::SequentialStream
    }
}

/// Detect the on-disk format and load weights to `device`.
///
/// A `.gguf` file uses the GGUF parser; a directory or `.safetensors` file uses
/// the safetensors path. Not GPU-gated — usable on a CPU device for testing.
pub fn load_weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<LoadedWeights> {
    let is_gguf = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if is_gguf {
        crate::gguf::load(path, dtype, device)
    } else if path.is_dir() || path.extension().is_some() {
        crate::safetensors::load(path, dtype, device)
    } else {
        Err(LoaderError::InvalidConfig(format!(
            "cannot determine format of {}",
            path.display()
        )))
    }
}

/// GPU strategy execution: load weights, account them in the global ledger, and
/// build a model via its factory.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::path::Path;

    use ayaka_memory::{MemoryPurpose, global_ledger};

    use super::*;
    use crate::device_map::DeviceMap;
    use crate::traits::{LoadConfig, ModelLoaderFactory, NormalModel};

    /// Load all weights into VRAM and register them as `Weights` in the ledger.
    pub fn full_gpu(
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
    ) -> Result<Box<dyn NormalModel>> {
        let weights = load_weights(path, config.dtype, &config.device)?;
        let bytes = weights.weight_bytes;
        let num_layers = weights.metadata.num_hidden_layers;
        global_ledger().register(MemoryPurpose::Weights, bytes);
        let device_map = DeviceMap::all_gpu(config.device.clone(), num_layers);
        let model = factory
            .load(weights, &device_map)
            .inspect_err(|_| {
                // Roll back the accounting if model construction fails.
                let _ = global_ledger().deregister(MemoryPurpose::Weights, bytes);
            })?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_full_gpu_when_model_fits_vram() {
        // 1 GiB model, 8 GiB VRAM, util 0.9 -> 7.2 GiB usable.
        let gib = 1024 * 1024 * 1024;
        let k = select_strategy(gib, 8 * gib, 16 * gib, 0.9);
        assert_eq!(k, StrategyKind::FullGpu);
    }

    #[test]
    fn selects_partial_when_model_fits_vram_plus_ram() {
        let gib = 1024 * 1024 * 1024;
        // 10 GiB model, 8 GiB VRAM (7.2 usable) + 16 GiB RAM -> partial.
        let k = select_strategy(10 * gib, 8 * gib, 16 * gib, 0.9);
        assert_eq!(k, StrategyKind::PartialOffload);
    }

    #[test]
    fn selects_sequential_when_model_exceeds_vram_plus_ram() {
        let gib = 1024 * 1024 * 1024;
        // 100 GiB model dwarfs 8 GiB VRAM + 16 GiB RAM.
        let k = select_strategy(100 * gib, 8 * gib, 16 * gib, 0.9);
        assert_eq!(k, StrategyKind::SequentialStream);
    }
}
