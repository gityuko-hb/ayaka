#pragma once

// Internal launcher for the Marlin-optimized W4A16 GEMM. Not installed.
// Called by native/src/ops/quant_gemm.cc when the caller opts in via a
// non-null workspace. `b_quant` must be in Marlin tile layout
// `[N/16, K/16, 16, 8]` (see `repack_to_marlin` on the Rust side and
// `marlin_shuffle.cuh` for the geometry).

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Marlin-style W4A16 GEMM for SM80+. Uses shared-memory double buffering of
// weight tiles via `cp.async`, register-level 4-bit unpacking via PTX `prmt`,
// and f32 FMA accumulation (Tensor Core `mma.sync` is a follow-up).
//
// Requirements (returns AYAKA_STATUS_UNSUPPORTED otherwise):
//   - compute capability major >= 80,
//   - out dtype F16 or BF16,
//   - N and K multiples of 16 (Marlin tile alignment),
//   - group_size a multiple of 16 (one scale per K-tile row).
AYAKA_NODISCARD ayaka_status_t w4a16_gemm_marlin(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& a,
    const ayaka_tensor_view_t& b_quant, const ayaka_tensor_view_t& b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* workspace, size_t workspace_bytes,
    ayaka_stream_t stream);

}  // namespace ayaka
