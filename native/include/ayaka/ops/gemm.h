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
 * Dense GEMM via cuBLASLt:
 *
 *   out = op_a(a) @ op_b(b) (+ bias)
 *
 * where op_x transposes its argument when trans_x is non-zero.
 *
 * Requirements:
 *   - out, a, b (and bias when non-NULL) are CUDA tensors on the same device.
 *   - All tensors share one dtype: F32, F16, or BF16. Accumulation is f32
 *     (CUBLAS_COMPUTE_32F; no TF32 down-conversion).
 *   - out, a, b have rank 2 and are row-major contiguous in their stored
 *     shape; transposition is logical via the trans flags:
 *       op_a(a) is [M, K]: a stored [M, K] (trans_a == 0) or [K, M].
 *       op_b(b) is [K, N]: b stored [K, N] (trans_b == 0) or [N, K].
 *       out is [M, N]; K must be >= 1.
 *   - bias, when non-NULL, has rank 1 and shape [N]; it is added to every
 *     row of out (cuBLASLt bias epilogue).
 *   - workspace is caller-owned device memory handed to cuBLASLt. NULL with
 *     workspace_bytes == 0 is valid; algorithm selection is then restricted
 *     to workspace-free algorithms.
 *
 * alpha/beta are fixed at 1/0: out is overwritten, never accumulated into.
 * The cuBLASLt handle is cached per device and reused across calls.
 *
 * The launch is asynchronous on `stream`; only setup/launch errors are
 * reported.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_gemm(const ayaka_tensor_view_t* out,
           const ayaka_tensor_view_t* a,
           const ayaka_tensor_view_t* b,
           const ayaka_tensor_view_t* bias,
           int32_t trans_a,
           int32_t trans_b,
           void* workspace,
           size_t workspace_bytes,
           ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
