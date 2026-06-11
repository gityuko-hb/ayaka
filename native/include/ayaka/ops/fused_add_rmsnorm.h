#pragma once

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fused residual-add + RMSNorm over the last dimension:
 *
 *   residual_out[row, i] = input[row, i] + residual[row, i]
 *   out[row, i] = residual_out[row, i]
 *                 * rsqrt(mean(residual_out[row, :]^2) + eps) * weight[i]
 *
 * Requirements:
 *   - All tensors are CUDA tensors on the same device, one dtype
 *     (F32/F16/BF16), row-major contiguous.
 *   - out, residual_out, input, residual share one shape (rank >= 1);
 *     weight is rank 1 matching the last dim.
 *   - In-place is supported and expected: out may alias input, and
 *     residual_out may alias residual. out must not alias residual_out.
 *   - The sum is accumulated in f32; the normalization pass reads the
 *     rounded residual_out values (one storage-dtype rounding step).
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_fused_add_rmsnorm(const ayaka_tensor_view_t* out,
                        const ayaka_tensor_view_t* residual_out,
                        const ayaka_tensor_view_t* input,
                        const ayaka_tensor_view_t* residual,
                        const ayaka_tensor_view_t* weight,
                        float eps,
                        ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
