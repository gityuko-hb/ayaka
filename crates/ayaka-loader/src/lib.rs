pub mod device_map;
pub mod error;
pub mod estimate;
pub mod gguf;
pub mod metadata;
pub mod safetensors;
pub mod sequential;
pub mod strategy;
pub mod traits;
pub mod weights;

// The live driver query and the profiler that uses it need CUDA.
#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(feature = "cuda")]
pub mod profiler;

pub use device_map::{DeviceMap, LayerPlacement};
pub use error::{LoaderError, Result};
pub use estimate::{MemoryEstimate, MemoryEstimator};
pub use metadata::{MlaConfig, ModelMetadata, MoeConfig, QuantConfig};
pub use sequential::SequentialStreamModel;
pub use strategy::{StrategyKind, peek_metadata, select_strategy};
pub use traits::{LoadConfig, ModelLoaderFactory, NormalModel, StreamableModel};
pub use weights::{LoadedQuantWeights, LoadedWeights, MmapWeights};

// CUDA-only surface: the isolated driver actor and the fallback orchestrator.
#[cfg(all(feature = "cuda", feature = "async"))]
pub use driver::driver_memory_info_async;
#[cfg(feature = "cuda")]
pub use driver::{CudaDriverActor, driver_memory_info, global_driver};
#[cfg(feature = "cuda")]
pub use strategy::gpu::load_with_fallback;
