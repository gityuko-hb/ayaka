//! Quantization schemes and GGUF block layouts.
//!
//! Owns the description of how weights are stored on disk and how many bytes
//! each stored weight costs. `ayaka-loader` uses [`QuantScheme::bytes_per_weight`]
//! for memory estimation and the [`GgufBlock`] descriptors to dequantize GGUF
//! tensors to F16/BF16 before they reach any kernel.

pub mod gguf_block;
pub mod kquants;
pub mod qtensor;
pub mod repack;
pub mod scheme;

pub use gguf_block::{GgufBlock, GgufDequantError, GgufDtype};
pub use qtensor::QTensor;
pub use repack::{RepackedWeights, WeightRepack, repack_to_marlin};
pub use scheme::QuantScheme;
