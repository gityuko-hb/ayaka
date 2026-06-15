#pragma once

#include <stdint.h>

#include "ayaka/kv_layout.h"
#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scatter new key/value tokens into a paged KV cache (the decode write path).
 *
 * For each token n with slot = slot_mapping[n] >= 0, the [H_kv, D] key and
 * value rows are copied verbatim (no dtype conversion) into the cache at the
 * (block, in-block position) addressed by slot. Tokens with slot < 0 are
 * skipped, so callers can pad slot_mapping with -1.
 *
 * Shapes / dtypes:
 *   - key:          [N, H_kv, D], F32/F16/BF16
 *   - value:        [N, H_kv, D], same dtype as key
 *   - kv_cache:     [num_blocks, 2, block_size, H_kv, D], same dtype as key;
 *                   dim 1 selects K (0) or V (1)
 *   - slot_mapping: [N], I64; each value is a flat slot index
 *                   (block * block_size + in-block position) or -1 to skip.
 *                   In-range slots are not validated on the host.
 *   - layout:       ayaka_kv_layout_t; only NHD is implemented (HND returns
 *                   AYAKA_STATUS_UNSUPPORTED).
 *   - all tensors contiguous, on one CUDA device.
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t ayaka_append_kv(
    const ayaka_tensor_view_t* key, const ayaka_tensor_view_t* value,
    const ayaka_tensor_view_t* kv_cache,
    const ayaka_tensor_view_t* slot_mapping, int32_t layout,
    ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif