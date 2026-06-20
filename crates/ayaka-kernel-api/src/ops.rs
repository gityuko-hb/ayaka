//! Thin wrappers over the raw `extern "C"` kernel entry points.
//!
//! These convert `ayaka_status_t` into `Result` but stay `unsafe`: the views
//! carry raw device pointers the type system cannot vouch for. The safe
//! surface lives one layer up in `ayaka-candle`, which derives every view
//! from a live tensor.

#[cfg(feature = "cuda")]
use crate::ffi::{self, AyakaStream, AyakaTensorView, RopeLayout};
#[cfg(feature = "cuda")]
use ayaka_core::layout::KvLayout;
#[cfg(feature = "cuda")]
use ayaka_error::AyakaError;

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
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_rmsnorm(out, input, weight, eps, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// Fused residual-add + RMSNorm: `residual_out = input + residual`,
/// `out = rmsnorm(residual_out) * weight`.
///
/// # Safety
///
/// - All views must describe live CUDA allocations matching their metadata
///   for the duration of the kernel execution.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` may alias `input`; `residual_out` may alias `residual`. No other
///   aliasing is allowed (the native side rejects `out == residual_out`).
#[cfg(feature = "cuda")]
pub unsafe fn fused_add_rmsnorm(
    out: &AyakaTensorView,
    residual_out: &AyakaTensorView,
    input: &AyakaTensorView,
    residual: &AyakaTensorView,
    weight: &AyakaTensorView,
    eps: f32,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_fused_add_rmsnorm(out, residual_out, input, residual, weight, eps, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// In-place rotary position embedding on `query` and `key`.
///
/// # Safety
///
/// - All views must describe live CUDA allocations matching their metadata
///   for the duration of the kernel execution; `query` and `key` are written.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - Every value in `positions` must be a valid row index into
///   `cos_sin_cache`; this is not validated on the host.
#[cfg(feature = "cuda")]
pub unsafe fn rope(
    query: &AyakaTensorView,
    key: &AyakaTensorView,
    positions: &AyakaTensorView,
    cos_sin_cache: &AyakaTensorView,
    layout: RopeLayout,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_rope(query, key, positions, cos_sin_cache, layout as i32, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// SwiGLU: `out[.., d] = silu(input[.., d]) * input[.., hidden + d]` with
/// `input.last_dim == 2 * out.last_dim`.
///
/// # Safety
///
/// - `out` and `input` must describe live CUDA allocations matching their
///   metadata for the duration of the kernel execution.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` must not alias `input`.
#[cfg(feature = "cuda")]
pub unsafe fn silu_and_mul(
    out: &AyakaTensorView,
    input: &AyakaTensorView,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_silu_and_mul(out, input, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// Dense GEMM via cuBLASLt: `out = op_a(a) @ op_b(b) (+ bias)` with f32
/// accumulation; `op_x` transposes its argument when `trans_x` is set.
///
/// # Safety
///
/// - `out`, `a`, `b`, and `bias` (when `Some`) must describe live CUDA
///   allocations that match their metadata for the duration of the kernel
///   execution, not just this call.
/// - `workspace` must be null with `workspace_bytes == 0`, or a live device
///   allocation of at least `workspace_bytes` on the views' device that no
///   concurrent work uses until this launch completes.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` must not alias `a`, `b`, `bias`, or `workspace`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    out: &AyakaTensorView,
    a: &AyakaTensorView,
    b: &AyakaTensorView,
    bias: Option<&AyakaTensorView>,
    trans_a: bool,
    trans_b: bool,
    workspace: *mut core::ffi::c_void,
    workspace_bytes: usize,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    let bias_ptr = bias.map_or(core::ptr::null(), |view| view as *const AyakaTensorView);
    unsafe {
        ffi::ayaka_gemm(
            out,
            a,
            b,
            bias_ptr,
            trans_a as i32,
            trans_b as i32,
            workspace,
            workspace_bytes,
            stream,
        )
    }
    .into_result()
    .map_err(AyakaError::from)
}

/// Quantized GEMM (W4A16): `out = a @ dequant(b_quant) (+ bias)` with f32
/// accumulation. The kernel dequantizes 4-bit weights on-the-fly using
/// per-group scales (and mins when provided).
///
/// # Safety
///
/// - `out`, `a`, `b_quant`, `b_scales` (and `b_mins`/`bias` when `Some`) must
///   describe live CUDA allocations matching their metadata for the duration
///   of the kernel execution.
/// - `workspace` must be null with `workspace_bytes == 0`, or a live device
///   allocation of at least `workspace_bytes`.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` must not alias any input or `workspace`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn quant_gemm(
    out: &AyakaTensorView,
    a: &AyakaTensorView,
    b_quant: &AyakaTensorView,
    b_scales: &AyakaTensorView,
    b_mins: Option<&AyakaTensorView>,
    bias: Option<&AyakaTensorView>,
    group_size: i32,
    workspace: *mut core::ffi::c_void,
    workspace_bytes: usize,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    let b_mins_ptr = b_mins.map_or(core::ptr::null(), |v| v as *const AyakaTensorView);
    let bias_ptr = bias.map_or(core::ptr::null(), |v| v as *const AyakaTensorView);
    unsafe {
        ffi::ayaka_quant_gemm(
            out,
            a,
            b_quant,
            b_scales,
            b_mins_ptr,
            bias_ptr,
            group_size,
            workspace,
            workspace_bytes,
            stream,
        )
    }
    .into_result()
    .map_err(AyakaError::from)
}

/// Marlin-layout W4A16 GEMM. Same contract as [`quant_gemm`] but `b_quant`
/// holds the Marlin tile layout `[N/16, K/16, 16, 8]`.
///
/// # Safety
///
/// Identical to [`quant_gemm`]: all views must describe live CUDA allocations
/// matching their metadata for the kernel's duration; `workspace` null with
/// `workspace_bytes == 0` or a live allocation; `stream` valid on the device;
/// `out` must not alias any input or `workspace`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn quant_gemm_marlin(
    out: &AyakaTensorView,
    a: &AyakaTensorView,
    b_quant: &AyakaTensorView,
    b_scales: &AyakaTensorView,
    b_mins: Option<&AyakaTensorView>,
    bias: Option<&AyakaTensorView>,
    group_size: i32,
    workspace: *mut core::ffi::c_void,
    workspace_bytes: usize,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    let b_mins_ptr = b_mins.map_or(core::ptr::null(), |v| v as *const AyakaTensorView);
    let bias_ptr = bias.map_or(core::ptr::null(), |v| v as *const AyakaTensorView);
    unsafe {
        ffi::ayaka_quant_gemm_marlin(
            out,
            a,
            b_quant,
            b_scales,
            b_mins_ptr,
            bias_ptr,
            group_size,
            workspace,
            workspace_bytes,
            stream,
        )
    }
    .into_result()
    .map_err(AyakaError::from)
}

/// Scatter `key`/`value` tokens into `kv_cache` in place (decode write path).
///
/// `layout` selects the KV cache memory layout; only NHD is implemented.
///
/// # Safety
///
/// - `key`, `value`, `kv_cache`, and `slot_mapping` must describe live CUDA
///   allocations matching their metadata for the duration of the kernel
///   execution; `kv_cache` is written.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `kv_cache` must not alias `key` or `value`. Slot values must be in range
///   or `-1` (not validated on the host).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn append_kv(
    key: &AyakaTensorView,
    value: &AyakaTensorView,
    kv_cache: &AyakaTensorView,
    slot_mapping: &AyakaTensorView,
    layout: KvLayout,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe { ffi::ayaka_append_kv(key, value, kv_cache, slot_mapping, layout as i32, stream) }
        .into_result()
        .map_err(AyakaError::from)
}

/// Paged attention, decode phase: `out = softmax(query @ K^T * scale) @ V`
/// over each sequence's paged KV cache. Only NHD `layout` is implemented.
///
/// # Safety
///
/// - `out`, `query`, `kv_cache`, `block_tables`, and `seq_lens` must describe
///   live CUDA allocations matching their metadata for the duration of the
///   kernel execution; `out` is written.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` must not alias any input. Block-table entries must be valid physical
///   block ids (not validated on the host).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode(
    out: &AyakaTensorView,
    query: &AyakaTensorView,
    kv_cache: &AyakaTensorView,
    block_tables: &AyakaTensorView,
    seq_lens: &AyakaTensorView,
    scale: f32,
    layout: KvLayout,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe {
        ffi::ayaka_paged_attention_decode(
            out,
            query,
            kv_cache,
            block_tables,
            seq_lens,
            scale,
            layout as i32,
            stream,
        )
    }
    .into_result()
    .map_err(AyakaError::from)
}

/// Split-KV (v2) paged attention decode for long contexts. Same contract as
/// [`paged_attention_decode`] plus caller-supplied scratch (`exp_sums`,
/// `max_logits`, `tmp_out`) sized to `ceil(max_seq_len / partition_size)`.
///
/// # Safety
///
/// - All views must describe live CUDA allocations matching their metadata for
///   the duration of the kernel execution; `out` and the scratch are written.
/// - `stream` must be a valid CUDA stream on the views' device.
/// - `out` and scratch must not alias any input.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn paged_attention_decode_v2(
    out: &AyakaTensorView,
    query: &AyakaTensorView,
    kv_cache: &AyakaTensorView,
    block_tables: &AyakaTensorView,
    seq_lens: &AyakaTensorView,
    exp_sums: &AyakaTensorView,
    max_logits: &AyakaTensorView,
    tmp_out: &AyakaTensorView,
    scale: f32,
    layout: KvLayout,
    stream: AyakaStream,
) -> Result<(), AyakaError> {
    unsafe {
        ffi::ayaka_paged_attention_decode_v2(
            out,
            query,
            kv_cache,
            block_tables,
            seq_lens,
            exp_sums,
            max_logits,
            tmp_out,
            scale,
            layout as i32,
            stream,
        )
    }
    .into_result()
    .map_err(AyakaError::from)
}

/// Tokens per split-KV partition (native constant). Always >= 1.
#[cfg(feature = "cuda")]
pub fn paged_attention_partition_size() -> i32 {
    unsafe { ffi::ayaka_paged_attention_partition_size() }
}
