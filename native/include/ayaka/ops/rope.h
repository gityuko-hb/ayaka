#pragma once

#include <stdint.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rotary pair layout within each head's leading rot_dim elements. */
typedef enum {
    /* Rotate-halves (GPT-NeoX / Llama): pairs (d, d + rot_dim/2). */
    AYAKA_ROPE_LAYOUT_NEOX = 0,
    /* Interleaved (GPT-J): pairs (2d, 2d + 1). */
    AYAKA_ROPE_LAYOUT_GPTJ = 1,
} ayaka_rope_layout_t;

/*
 * In-place rotary position embedding applied to query and key.
 *
 * For each token t and head h, the leading rot_dim elements of the head are
 * rotated using row positions[t] of cos_sin_cache; elements beyond rot_dim
 * pass through unchanged.
 *
 * Shapes / dtypes:
 *   - query:         [num_tokens, num_heads,    head_dim], F32/F16/BF16
 *   - key:           [num_tokens, num_kv_heads, head_dim], same dtype as query
 *   - positions:     [num_tokens], I64; every value must be a valid row of
 *                    cos_sin_cache (not validated on the host)
 *   - cos_sin_cache: [max_position, rot_dim], F32; row p = rot_dim/2 cosines
 *                    followed by rot_dim/2 sines for position p
 *   - rot_dim must be even and <= head_dim; all tensors contiguous, on one
 *     CUDA device. Rotation math runs in f32.
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API ayaka_status_t
ayaka_rope(const ayaka_tensor_view_t* query,
           const ayaka_tensor_view_t* key,
           const ayaka_tensor_view_t* positions,
           const ayaka_tensor_view_t* cos_sin_cache,
           int32_t layout,
           ayaka_stream_t stream);

#ifdef __cplusplus
}
#endif
