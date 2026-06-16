//! Per-layer device placement produced by a load strategy.

use candle_core::Device;

/// Where a single decoder layer's weights live.
#[derive(Debug, Clone)]
pub enum LayerPlacement {
    /// Resident on the GPU for the whole run.
    Gpu,
    /// Kept in host RAM and staged to the GPU on demand (partial offload).
    Cpu,
    /// Streamed: mmap'd, loaded just before use, and dropped after.
    Streamed,
}

/// The placement decision for every layer plus the compute device.
#[derive(Debug, Clone)]
pub struct DeviceMap {
    /// Compute device (where forward runs).
    pub device: Device,
    /// One entry per decoder layer, in order.
    pub layers: Vec<LayerPlacement>,
}

impl DeviceMap {
    /// All layers resident on `device`.
    pub fn all_gpu(
        device: Device,
        num_layers: usize,
    ) -> Self {
        Self {
            device,
            layers: vec![LayerPlacement::Gpu; num_layers],
        }
    }

    /// First `n_gpu_layers` on GPU, the rest on CPU (partial offload).
    pub fn partial(
        device: Device,
        num_layers: usize,
        n_gpu_layers: usize,
    ) -> Self {
        let n_gpu = n_gpu_layers.min(num_layers);
        let layers = (0..num_layers)
            .map(|i| {
                if i < n_gpu {
                    LayerPlacement::Gpu
                } else {
                    LayerPlacement::Cpu
                }
            })
            .collect();
        Self { device, layers }
    }

    /// Every layer streamed one at a time.
    pub fn streamed(
        device: Device,
        num_layers: usize,
    ) -> Self {
        Self {
            device,
            layers: vec![LayerPlacement::Streamed; num_layers],
        }
    }

    /// Count of layers resident on the GPU.
    pub fn gpu_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerPlacement::Gpu))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_split_is_clamped() {
        let m = DeviceMap::partial(Device::Cpu, 10, 4);
        assert_eq!(m.gpu_layer_count(), 4);
        assert!(matches!(m.layers[3], LayerPlacement::Gpu));
        assert!(matches!(m.layers[4], LayerPlacement::Cpu));

        let all = DeviceMap::partial(Device::Cpu, 10, 99);
        assert_eq!(all.gpu_layer_count(), 10);
    }
}
