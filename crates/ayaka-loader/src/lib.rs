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
pub use metadata::{MlaConfig, ModelMetadata, MoeConfig};
pub use sequential::SequentialStreamModel;
pub use strategy::{StrategyKind, select_strategy};
pub use traits::{LoadConfig, ModelLoaderFactory, NormalModel, StreamableModel};
pub use weights::LoadedWeights;
