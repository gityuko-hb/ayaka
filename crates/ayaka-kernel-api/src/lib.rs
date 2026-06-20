pub mod error;
pub mod ffi;
pub mod ops;

#[cfg(feature = "cuda")]
pub mod mem;

pub use error::KernelError;
pub use ffi::*;
#[cfg(feature = "cuda")]
pub use ops::{
    append_kv, fused_add_rmsnorm, gemm, paged_attention_decode, paged_attention_decode_v2,
    paged_attention_partition_size, quant_gemm, quant_gemm_marlin, rmsnorm, rope, silu_and_mul,
};
