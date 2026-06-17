//! Load-strategy selection and (GPU-only) execution.
//!
//! The selection math and format dispatch are pure and CPU-testable; the
//! strategies that actually move weights to VRAM are gated behind `cuda`.
//!
//! ## Graceful degradation (GIAI ĐOẠN 0 — Bước 0.2)
//! [`select_strategy`] picks a starting rung from a static size comparison.
//! [`gpu::load_with_fallback`] then executes it and, on a VRAM-exhaustion error
//! ([`LoaderError::is_vram_oom`]), walks down the ladder
//! `FullGpu → PartialOffload → SequentialStream` via [`StrategyKind::degrade`]
//! instead of panicking. The bottom rung surfaces the original error.

use std::fs;
use std::path::Path;

use candle_core::{DType, Device};

use crate::error::{LoaderError, Result};
use crate::metadata::ModelMetadata;
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

impl StrategyKind {
    /// The next, more conservative rung in the degradation ladder, or `None` if
    /// already at the bottom (`SequentialStream`).
    ///
    /// `FullGpu → PartialOffload → SequentialStream → None`
    pub fn degrade(self) -> Option<Self> {
        match self {
            StrategyKind::FullGpu => Some(StrategyKind::PartialOffload),
            StrategyKind::PartialOffload => Some(StrategyKind::SequentialStream),
            StrategyKind::SequentialStream => None,
        }
    }
}

/// Choose a strategy from the decision matrix compare the
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

/// `true` when `path` names a `.gguf` file (case-insensitive).
fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
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
    if is_gguf(path) {
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

/// Parse just the [`ModelMetadata`] without materializing any weights, so the
/// orchestrator can estimate footprint cheaply before committing to a load.
///
/// GGUF: mmap + parse the header/KV store only. Safetensors: read the sibling
/// or in-directory `config.json`.
pub fn peek_metadata(path: &Path) -> Result<ModelMetadata> {
    if is_gguf(path) {
        let file = fs::File::open(path).map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: read-only mmap of an immutable file, used only to parse the
        // GGUF header/metadata/tensor table (no tensor data is dereferenced
        // here). The mapping is dropped at the end of this function.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        crate::gguf::GgufFile::parse(&mmap)?.model_metadata()
    } else {
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join("config.json")
        };
        let json = fs::read_to_string(&config_path).map_err(|source| LoaderError::Io {
            path: config_path.clone(),
            source,
        })?;
        ModelMetadata::from_config_json(&json)
    }
}

/// GPU strategy execution: load weights, account them in the global ledger, and
/// build a model via its factory. Includes the reactive fallback orchestrator.
#[cfg(feature = "cuda")]
pub mod gpu {
    use ayaka_memory::{MemoryPurpose, global_ledger};
    use ayaka_quant::QuantScheme;

    use super::*;
    use crate::device_map::DeviceMap;
    use crate::estimate::MemoryEstimator;
    use crate::traits::{LoadConfig, ModelLoaderFactory, NormalModel};

    /// Register `resident_bytes` as `Weights` in the ledger, build the model via
    /// `factory`, and roll the accounting back if construction fails.
    fn finish_load(
        weights: LoadedWeights,
        device_map: DeviceMap,
        resident_bytes: usize,
        factory: &dyn ModelLoaderFactory,
    ) -> Result<Box<dyn NormalModel>> {
        global_ledger().register(MemoryPurpose::Weights, resident_bytes);
        factory
            .load(weights, &device_map)
            .inspect_err(|_| {
                let _ = global_ledger().deregister(MemoryPurpose::Weights, resident_bytes);
            })
    }

    /// Fractions of total weight bytes attributable to the fixed tensors and to
    /// one dense decoder layer. These ratios are dtype-independent, so we use an
    /// `F16` scheme purely to count parameters and then scale by the *actual*
    /// resident `weight_bytes` — correct for any compute dtype.
    fn weight_fractions(meta: &ModelMetadata) -> (f64, f64) {
        let est = MemoryEstimator::new(meta, QuantScheme::F16, 2, 0.9);
        let e = est.estimate();
        let total = e.total_weight_bytes.max(1) as f64;
        (
            e.fixed_bytes as f64 / total,
            e.per_layer_bytes as f64 / total,
        )
    }

    /// Load all weights into VRAM and register them as `Weights` in the ledger.
    pub fn full_gpu(
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
    ) -> Result<Box<dyn NormalModel>> {
        let weights = load_weights(path, config.dtype, &config.device)?;
        let resident = weights.weight_bytes;
        let num_layers = weights.metadata.num_hidden_layers;
        let device_map = DeviceMap::all_gpu(config.device.clone(), num_layers);
        finish_load(weights, device_map, resident, factory)
    }

    /// Keep as many layers resident as VRAM allows (auto-fit) and stage the rest
    /// from host RAM. The resident byte count registered in the ledger reflects
    /// only the GPU-resident portion.
    pub fn partial_offload(
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
    ) -> Result<Box<dyn NormalModel>> {
        let weights = load_weights(path, config.dtype, &config.device)?;
        let num_layers = weights.metadata.num_hidden_layers;

        // Auto-fit the GPU-resident layer count from live VRAM. `reserve = 0`
        // here; Phase 1.2 will pass the profiled KV + activation reservation.
        let info = crate::driver::driver_memory_info(&config.device)?;
        let n_gpu = {
            let est = MemoryEstimator::new(
                &weights.metadata,
                QuantScheme::F16,
                config.dtype.size_in_bytes(),
                config.gpu_memory_utilization,
            );
            est.fit_gpu_layers(info.free_bytes, info.total_bytes, 0)
        };

        let (fixed_frac, per_layer_frac) = weight_fractions(&weights.metadata);
        let resident =
            (weights.weight_bytes as f64 * (fixed_frac + per_layer_frac * n_gpu as f64)) as usize;

        let device_map = DeviceMap::partial(config.device.clone(), num_layers, n_gpu);
        finish_load(weights, device_map, resident, factory)
    }

    /// Stream layers one at a time from disk. Only the fixed tensors plus a
    /// single decoder layer are counted as resident.
    pub fn sequential_stream(
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
    ) -> Result<Box<dyn NormalModel>> {
        let weights = load_weights(path, config.dtype, &config.device)?;
        let num_layers = weights.metadata.num_hidden_layers;

        let (fixed_frac, per_layer_frac) = weight_fractions(&weights.metadata);
        let resident = (weights.weight_bytes as f64 * (fixed_frac + per_layer_frac)) as usize;

        let device_map = DeviceMap::streamed(config.device.clone(), num_layers);
        finish_load(weights, device_map, resident, factory)
    }

    /// Dispatch one rung of the ladder.
    fn execute_strategy(
        kind: StrategyKind,
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
    ) -> Result<Box<dyn NormalModel>> {
        match kind {
            StrategyKind::FullGpu => full_gpu(path, factory, config),
            StrategyKind::PartialOffload => partial_offload(path, factory, config),
            StrategyKind::SequentialStream => sequential_stream(path, factory, config),
        }
    }

    /// Best-effort proactive starting rung: estimate the resident weight
    /// footprint from metadata (no full load) and compare against total VRAM +
    /// host RAM. Returns `None` (→ caller defaults to `FullGpu`, then the
    /// reactive ladder corrects) when the resident dtype has no `QuantScheme`
    /// or metadata/driver probing fails.
    fn plan_initial_strategy(
        path: &Path,
        config: &LoadConfig,
        host_ram_bytes: usize,
    ) -> Option<StrategyKind> {
        let scheme = match config.dtype {
            DType::F16 => QuantScheme::F16,
            DType::BF16 => QuantScheme::BF16,
            _ => return None,
        };
        let meta = peek_metadata(path).ok()?;
        let est = MemoryEstimator::new(
            &meta,
            scheme,
            config.dtype.size_in_bytes(),
            config.gpu_memory_utilization,
        );
        let weight_bytes = est.estimate().total_weight_bytes;
        let info = crate::driver::driver_memory_info(&config.device).ok()?;
        Some(select_strategy(
            weight_bytes,
            info.total_bytes,
            host_ram_bytes,
            config.gpu_memory_utilization,
        ))
    }

    /// Load a model, degrading the strategy on VRAM exhaustion instead of
    /// failing hard. Returns the strategy that actually succeeded alongside the
    /// model. Non-OOM errors are propagated unchanged (never masked by a retry).
    pub fn load_with_fallback(
        path: &Path,
        factory: &dyn ModelLoaderFactory,
        config: &LoadConfig,
        host_ram_bytes: usize,
    ) -> Result<(StrategyKind, Box<dyn NormalModel>)> {
        let mut current =
            plan_initial_strategy(path, config, host_ram_bytes).unwrap_or(StrategyKind::FullGpu);
        loop {
            match execute_strategy(current, path, factory, config) {
                Ok(model) => return Ok((current, model)),
                Err(e) if e.is_vram_oom() => match current.degrade() {
                    Some(next) => {
                        tracing::warn!(
                            from = ?current,
                            to = ?next,
                            error = %e,
                            "VRAM exhausted; degrading load strategy"
                        );
                        current = next;
                    },
                    None => return Err(e),
                },
                Err(e) => return Err(e),
            }
        }
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

    #[test]
    fn degrade_walks_the_full_ladder_then_stops() {
        assert_eq!(
            StrategyKind::FullGpu.degrade(),
            Some(StrategyKind::PartialOffload)
        );
        assert_eq!(
            StrategyKind::PartialOffload.degrade(),
            Some(StrategyKind::SequentialStream)
        );
        assert_eq!(StrategyKind::SequentialStream.degrade(), None);
    }
}
