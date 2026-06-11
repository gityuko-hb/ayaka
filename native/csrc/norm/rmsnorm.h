#pragma once

// Internal launcher contract between the host-side C ABI wrapper
// (src/ops/rmsnorm.cc) and the CUDA translation unit (csrc/norm/rmsnorm.cu).
// Not installed; CUDA types stay out of this header so .cc files build
// without the CUDA toolkit headers.

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller: CUDA device, matching dtypes
// (F32/F16/BF16), contiguous, weight rank 1 matching the last input dim.
ayaka_status_t rmsnorm_cuda(const ayaka_tensor_view_t& out,
                            const ayaka_tensor_view_t& input,
                            const ayaka_tensor_view_t& weight,
                            float eps,
                            ayaka_stream_t stream);

}  // namespace ayaka
