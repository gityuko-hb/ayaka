//! Thin wrappers over the raw `extern "C"` kernel entry points.
//!
//! These convert `ayaka_status_t` into `Result` but stay `unsafe`: the views
//! carry raw device pointers the type system cannot vouch for. The safe
//! surface lives one layer up in `ayaka-candle`, which derives every view
//! from a live tensor.

#[cfg(feature = "cuda")]
use crate::error::KernelError;
#[cfg(feature = "cuda")]
use crate::ffi::{self, AyakaStream, AyakaTensorView};

/// RMSNorm over the last dimension: `out = input * rsqrt(mean(input^2) + eps) * weight`.
///
/// # Safety
///
/// - `out`, `input`, and `weight` must describe live CUDA allocations that
///   match their metadata (dtype, shape, strides, device) for the duration of
///   the kernel execution, not just this call.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` must not alias `weight`. `out == input` (in-place) is allowed.
#[cfg(feature = "cuda")]
pub unsafe fn rmsnorm(
    out: &AyakaTensorView,
    input: &AyakaTensorView,
    weight: &AyakaTensorView,
    eps: f32,
    stream: AyakaStream,
) -> Result<(), KernelError> {
    unsafe { ffi::ayaka_rmsnorm(out, input, weight, eps, stream) }.into_result()
}
