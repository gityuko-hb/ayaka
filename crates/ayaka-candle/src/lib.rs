//! Candle ↔ ayaka kernel bridge.
//!
//! Converts `candle_core::Tensor`s into the flat `TensorView` descriptors the
//! native C ABI understands and exposes the kernels as tensor ops. Candle
//! types stop at this crate: nothing below `ayaka-kernel-api` ever sees them.

#[cfg(feature = "cuda")]
pub mod extract;
pub mod ops;
