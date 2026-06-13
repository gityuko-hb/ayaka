#pragma once

// Internal contract between the C ABI wrapper (src/ops/gemm.cc) and the
// cuBLASLt backend translation unit (src/backends/cublaslt_gemm.cc).
// Not installed; cuBLASLt types stay out of this header.

#include <stddef.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller: rank-2 CUDA tensors on one device,
// matching dtypes (F32/F16/BF16), row-major contiguous, shapes consistent
// with the trans flags, optional rank-1 bias of length N, K >= 1.
AYAKA_NODISCARD ayaka_status_t gemm_cublaslt(const ayaka_tensor_view_t& out,
                                             const ayaka_tensor_view_t& a,
                                             const ayaka_tensor_view_t& b,
                                             const ayaka_tensor_view_t* bias,
                                             bool trans_a,
                                             bool trans_b,
                                             void* workspace,
                                             size_t workspace_bytes,
                                             ayaka_stream_t stream);

}  // namespace ayaka
