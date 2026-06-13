#pragma once

// Internal launcher contract between src/ops/rope.cc and csrc/pos/rope.cu.
// Not installed; no CUDA types.

#include <stdint.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller (see ayaka_rope).
AYAKA_NODISCARD ayaka_status_t
rope_cuda(const ayaka_tensor_view_t &query, const ayaka_tensor_view_t &key,
          const ayaka_tensor_view_t &positions,
          const ayaka_tensor_view_t &cos_sin_cache, int32_t layout,
          ayaka_stream_t stream);

} // namespace ayaka
