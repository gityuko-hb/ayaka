//! Typestate model pipeline.
//!
//! A [`ModelPipeline`] moves through `Unloaded → Mapped → Ready`. Each
//! transition consumes `self` and returns the next state, so calling
//! [`ModelPipeline::forward`] before the model is loaded is a *compile* error,
//! not a runtime panic.
//!
//! ```
//! use ayaka_pipeline::ModelPipeline;
//! use ayaka_loader::{DeviceMap, LoadConfig};
//! use candle_core::Device;
//!
//! let pipeline = ModelPipeline::new(LoadConfig::default())
//!     .map_devices(DeviceMap::all_gpu(Device::Cpu, 1));
//! // `pipeline` is in the `Mapped` state; `.forward(..)` is not callable yet.
//! assert_eq!(pipeline.device_map().layers.len(), 1);
//! ```
//!
//! Calling `forward` on a non-`Ready` pipeline does not compile:
//!
//! ```compile_fail
//! use ayaka_pipeline::ModelPipeline;
//! use ayaka_loader::LoadConfig;
//!
//! let pipeline = ModelPipeline::new(LoadConfig::default());
//! // ERROR: no method `forward` on `ModelPipeline<Unloaded>`.
//! let _ = pipeline.forward(todo!(), 0);
//! ```

use std::marker::PhantomData;

use candle_core::Tensor;

use ayaka_loader::{DeviceMap, LoadConfig, NormalModel};

/// Typestate markers.
pub mod state {
    /// No weights loaded, no device map.
    pub struct Unloaded;
    /// Device placement decided, weights not yet loaded.
    pub struct Mapped;
    /// Model loaded and ready to run.
    pub struct Ready;
}

use state::{Mapped, Ready, Unloaded};

/// A model load/run pipeline parameterized by its lifecycle state `S`.
pub struct ModelPipeline<S> {
    config: LoadConfig,
    device_map: Option<DeviceMap>,
    model: Option<Box<dyn NormalModel>>,
    _state: PhantomData<S>,
}

impl ModelPipeline<Unloaded> {
    /// Start a new pipeline in the `Unloaded` state.
    pub fn new(config: LoadConfig) -> Self {
        Self {
            config,
            device_map: None,
            model: None,
            _state: PhantomData,
        }
    }

    /// Decide device placement, advancing to `Mapped`.
    pub fn map_devices(
        self,
        device_map: DeviceMap,
    ) -> ModelPipeline<Mapped> {
        ModelPipeline {
            config: self.config,
            device_map: Some(device_map),
            model: None,
            _state: PhantomData,
        }
    }

    pub fn config(&self) -> &LoadConfig {
        &self.config
    }
}

impl ModelPipeline<Mapped> {
    /// Attach the loaded model, advancing to `Ready`.
    pub fn load(
        self,
        model: Box<dyn NormalModel>,
    ) -> ModelPipeline<Ready> {
        ModelPipeline {
            config: self.config,
            device_map: self.device_map,
            model: Some(model),
            _state: PhantomData,
        }
    }

    /// The device map chosen for this pipeline.
    pub fn device_map(&self) -> &DeviceMap {
        self.device_map
            .as_ref()
            .expect("Mapped pipeline always has a device map")
    }
}

impl ModelPipeline<Ready> {
    /// Run a forward pass. Only available once the model is loaded.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        self.model
            .as_ref()
            .expect("Ready pipeline always has a model")
            .forward(input_ids, seqlen_offset)
    }

    /// Borrow the loaded model.
    pub fn model(&self) -> &dyn NormalModel {
        self.model
            .as_ref()
            .expect("Ready pipeline always has a model")
            .as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    struct ConstModel {
        device: Device,
    }

    impl NormalModel for ConstModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
        ) -> candle_core::Result<Tensor> {
            // Echo the input shape as logits over a 1-token vocab.
            let (b, s) = input_ids.dims2()?;
            Tensor::zeros((b, s, 1), candle_core::DType::F32, &self.device)
        }
        fn device(&self) -> &Device {
            &self.device
        }
    }

    #[test]
    fn full_lifecycle_unloaded_to_ready() {
        let pipeline = ModelPipeline::new(LoadConfig::default())
            .map_devices(DeviceMap::all_gpu(Device::Cpu, 2))
            .load(Box::new(ConstModel {
                device: Device::Cpu,
            }));

        let input = Tensor::zeros((1, 4), candle_core::DType::U32, &Device::Cpu).unwrap();
        let out = pipeline.forward(&input, 0).unwrap();
        assert_eq!(out.dims(), &[1, 4, 1]);
    }
}
