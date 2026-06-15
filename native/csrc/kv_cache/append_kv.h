#pragma once

// Internal launcher contract between src/ops/append_kv.cc and
// csrc/kv_cache/append_kv.cu. Not installed; no CUDA types.

#include <stdint.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller (see ayaka_append_kv). Returns
// AYAKA_STATUS_UNSUPPORTED for layouts other than NHD.
AYAKA_NODISCARD ayaka_status_t
append_kv_cuda(const ayaka_tensor_view_t& key, const ayaka_tensor_view_t& value,
               const ayaka_tensor_view_t& kv_cache,
               const ayaka_tensor_view_t& slot_mapping, int32_t layout,
               ayaka_stream_t stream);

}  // namespace ayaka