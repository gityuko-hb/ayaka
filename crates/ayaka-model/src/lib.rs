//! Concrete model architectures for ayaka.
//!
//! Currently a Qwen3 reference implementation that exercises the loader's
//! load / memory / streaming mechanics. Models implement the traits in
//! `ayaka-loader` and register a factory into `ayaka-registry`.

pub mod config;
pub mod model;
pub mod quant_model;

use std::sync::Arc;

use ayaka_loader::{
    DeviceMap, LoadedQuantWeights, LoadedWeights, ModelLoaderFactory, NormalModel, Result,
};

pub use config::Qwen3Config;
pub use model::{DecoderLayer, Qwen3Model, Qwen3Shared, Qwen3Streamer};
pub use quant_model::{QuantDecoderLayer, Qwen3QuantModel, Qwen3QuantStreamer};

/// Builds a [`Qwen3Model`] from loaded weights.
pub struct Qwen3Factory;

impl ModelLoaderFactory for Qwen3Factory {
    fn arch_id(&self) -> &'static str {
        "qwen3"
    }

    fn load(
        &self,
        weights: LoadedWeights,
        device_map: &DeviceMap,
    ) -> Result<Box<dyn NormalModel>> {
        let cfg = Qwen3Config::from_metadata(&weights.metadata);
        let dtype = weights.vb.dtype();
        let device = device_map.device.clone();
        let model = Qwen3Model::load(&weights.vb, cfg, dtype, &device)?;
        Ok(Box::new(model))
    }
}

/// Builds a [`Qwen3QuantModel`] from quantized weights, or delegates float
/// loading to [`Qwen3Factory`]. Same `arch_id` as [`Qwen3Factory`] — register
/// this one instead when quantized loading is available.
pub struct Qwen3QuantFactory;

impl ModelLoaderFactory for Qwen3QuantFactory {
    fn arch_id(&self) -> &'static str {
        "qwen3"
    }

    fn load(
        &self,
        weights: LoadedWeights,
        device_map: &DeviceMap,
    ) -> Result<Box<dyn NormalModel>> {
        Qwen3Factory.load(weights, device_map)
    }

    fn load_quantized(
        &self,
        weights: LoadedQuantWeights,
        device_map: &DeviceMap,
    ) -> Result<Box<dyn NormalModel>> {
        let cfg = Qwen3Config::from_metadata(&weights.metadata);
        let dtype = weights.vb.dtype();
        let device = device_map.device.clone();
        let model = Qwen3QuantModel::load_quantized(
            &weights.vb,
            &weights.qtensors,
            &weights.repacked,
            cfg,
            dtype,
            &device,
        )?;
        Ok(Box::new(model))
    }
}

/// Register Qwen3 in the global model registry.
pub fn register() {
    ayaka_registry::register(Arc::new(Qwen3Factory));
}
