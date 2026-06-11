#pragma once

// Internal launcher contract between src/ops/silu_and_mul.cc and
// csrc/activation/swiglu.cu. Not installed; no CUDA types.

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller (see ayaka_silu_and_mul).
ayaka_status_t silu_and_mul_cuda(const ayaka_tensor_view_t& out,
                                 const ayaka_tensor_view_t& input,
                                 ayaka_stream_t stream);

}  // namespace ayaka
