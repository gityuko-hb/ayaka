//! Concrete model architectures for ayaka.
//!
//! Currently a Qwen3 reference implementation that exercises the loader's
//! load / memory / streaming mechanics. Models implement the traits in
//! `ayaka-loader` and register a factory into `ayaka-registry`.

pub mod config;
pub mod model;

use std::sync::Arc;

use ayaka_loader::{DeviceMap, LoadedWeights, ModelLoaderFactory, NormalModel, Result};

pub use config::Qwen3Config;
pub use model::{DecoderLayer, Qwen3Model, Qwen3Shared, Qwen3Streamer};

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

/// Register Qwen3 in the global model registry.
pub fn register() {
    ayaka_registry::register(Arc::new(Qwen3Factory));
}
