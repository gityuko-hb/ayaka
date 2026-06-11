#pragma once

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Root-mean-square layer normalization over the last dimension:
 *
 *   out[row, i] = input[row, i] * rsqrt(mean(input[row, :]^2) + eps) * weight[i]
 *
 * Requirements:
 *   - out, input, weight are CUDA tensors on the same device.
 *   - out, input, weight share one dtype: F32, F16, or BF16.
 *   - input rank >= 1; out shape == input shape; weight rank == 1 with
 *     weight.shape[0] == input.shape[rank - 1].
 *   - All tensors row-major contiguous.
 *   - Accumulation is f32 regardless of storage dtype.
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_rmsnorm(const ayaka_tensor_view_t* out,
              const ayaka_tensor_view_t* input,
              const ayaka_tensor_view_t* weight,
              float eps,
              ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
