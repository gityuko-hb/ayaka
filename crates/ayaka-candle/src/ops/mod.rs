pub mod append_kv;
pub mod fused_add_rmsnorm;
pub mod gemm;
pub mod paged_attention;
pub mod quant_gemm;
pub mod rmsnorm;
pub mod rope;
pub mod silu_and_mul;

#[cfg(feature = "cuda")]
pub use append_kv::append_kv;
pub use append_kv::{KvLayout, append_kv_ref};
pub use fused_add_rmsnorm::fused_add_rmsnorm_ref;
#[cfg(feature = "cuda")]
pub use fused_add_rmsnorm::{fused_add_rmsnorm, fused_add_rmsnorm_new};
pub use gemm::gemm_ref;
#[cfg(feature = "cuda")]
pub use gemm::{gemm, gemm_new};
#[cfg(feature = "cuda")]
pub use paged_attention::{
    paged_attention_decode, paged_attention_decode_auto, paged_attention_decode_new,
    paged_attention_decode_v2, paged_attention_partition_size,
};
pub use paged_attention::{paged_attention_decode_partitioned_ref, paged_attention_decode_ref};
pub use quant_gemm::{QuantLinear, quant_gemm_ref};
#[cfg(feature = "cuda")]
pub use quant_gemm::{quant_gemm, quant_gemm_marlin, quant_gemm_marlin_new, quant_gemm_new};
pub use rmsnorm::rmsnorm_ref;
#[cfg(feature = "cuda")]
pub use rmsnorm::{rmsnorm, rmsnorm_new};
#[cfg(feature = "cuda")]
pub use rope::rope;
pub use rope::{RopeLayout, cos_sin_cache, rope_ref};
pub use silu_and_mul::silu_and_mul_ref;
#[cfg(feature = "cuda")]
pub use silu_and_mul::{silu_and_mul, silu_and_mul_new};
