#pragma once

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SwiGLU activation over a gate/up projection packed in the last dimension:
 *
 *   out[row, d] = silu(input[row, d]) * input[row, hidden + d]
 *
 * where input's last dim is 2 * hidden and out's last dim is hidden.
 *
 * Requirements:
 *   - out and input are CUDA tensors on the same device, one dtype
 *     (F32/F16/BF16), row-major contiguous.
 *   - out and input share all leading dims; input.last_dim == 2 * out.last_dim.
 *   - silu runs in f32 regardless of storage dtype.
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API ayaka_status_t
ayaka_silu_and_mul(const ayaka_tensor_view_t* out,
                   const ayaka_tensor_view_t* input,
                   ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
