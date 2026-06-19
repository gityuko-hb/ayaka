#pragma once

#include <stddef.h>
#include <stdint.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Quantized GEMM (W4A16): out = a @ dequant(b_quant) (+ bias)
 *
 * Activations `a` are F16/BF16; weights `b_quant` are packed 4-bit quants
 * (U8 tensor); `b_scales` are per-group F16 scales. The kernel dequantizes
 * weights on-the-fly during the matrix multiply.
 *
 * Parameters:
 *   - out:       [M, N] F16/BF16 output tensor.
 *   - a:         [M, K] F16/BF16 activation tensor (row-major contiguous).
 *   - b_quant:   [N, K/2] U8 packed 4-bit quants (2 weights per byte,
 *                low nibble = even index, high nibble = odd index).
 *   - b_scales:  [N, K/group_size] F16 per-group scales.
 *   - b_mins:    [N, K/group_size] F16 per-group mins, or NULL if the
 *                quantization scheme has no mins (e.g. Q4_0).
 *   - bias:      [N] F16/BF16 bias, or NULL.
 *   - group_size: Number of weights sharing one scale (32 for Q4_K sub-blocks,
 *                64/128 for AWQ/GPTQ).
 *   - workspace: Caller-owned device memory, or NULL with workspace_bytes == 0.
 *   - stream:    CUDA stream for async launch.
 *
 * Requirements:
 *   - All tensors are CUDA tensors on the same device.
 *   - out, a, b_quant, b_scales (and bias/mins when non-NULL) are contiguous.
 *   - K must be a multiple of group_size * 2 (so K/2 is a multiple of
 * group_size).
 *   - Accumulation is F32; output is cast to out dtype.
 *
 * The launch is asynchronous on `stream`.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t ayaka_quant_gemm(
    const ayaka_tensor_view_t* out, const ayaka_tensor_view_t* a,
    const ayaka_tensor_view_t* b_quant, const ayaka_tensor_view_t* b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* workspace, size_t workspace_bytes,
    ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
