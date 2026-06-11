pub mod error;
pub mod ffi;
pub mod ops;

pub use error::KernelError;
pub use ffi::*;
#[cfg(feature = "cuda")]
pub use ops::{fused_add_rmsnorm, rmsnorm, rope, silu_and_mul};
