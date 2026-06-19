#pragma once

// Internal launcher for W4A16 quantized GEMM. Not installed.
// Called by native/src/ops/quant_gemm.cc after validation.

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Naive W4A16 GEMM: out = a @ dequant(b_quant) (+ bias)
// Dispatches on out dtype (F16/BF16). b_quant is U8, b_scales/b_mins/bias are
// same dtype as out. b_mins and bias may be nullptr.
AYAKA_NODISCARD ayaka_status_t w4a16_gemm_naive(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& a,
    const ayaka_tensor_view_t& b_quant, const ayaka_tensor_view_t& b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* workspace, size_t workspace_bytes,
    ayaka_stream_t stream);

}  // namespace ayaka
