#pragma once

// Internal launcher contract between src/ops/fused_add_rmsnorm.cc and
// csrc/norm/fused_add_rmsnorm.cu. Not installed; no CUDA types.

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller (see ayaka_fused_add_rmsnorm).
ayaka_status_t fused_add_rmsnorm_cuda(const ayaka_tensor_view_t& out,
                                      const ayaka_tensor_view_t& residual_out,
                                      const ayaka_tensor_view_t& input,
                                      const ayaka_tensor_view_t& residual,
                                      const ayaka_tensor_view_t& weight,
                                      float eps,
                                      ayaka_stream_t stream);

}  // namespace ayaka
