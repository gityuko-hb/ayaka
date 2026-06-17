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
 * Paged attention, decode phase: one query token per sequence attends over its
 * cached K/V. Output is written to `out`.
 *
 * Shapes / dtypes:
 *   - out:          [num_seqs, num_heads, head_dim], F32/F16/BF16
 *   - query:        [num_seqs, num_heads, head_dim], same dtype as out
 *   - kv_cache:     [num_blocks, 2, block_size, H_kv, head_dim], same dtype;
 *                   dim 1 selects K (0) or V (1) (NHD layout)
 *   - block_tables: [num_seqs, max_num_blocks], I64; logical block -> physical
 *                   block id for each sequence
 *   - seq_lens:     [num_seqs], I64; number of cached tokens per sequence
 *   - scale:        softmax scale (typically 1/sqrt(head_dim))
 *   - layout:       ayaka_kv_layout_t; only NHD is implemented (HND returns
 *                   AYAKA_STATUS_UNSUPPORTED)
 *   - num_heads must be a multiple of H_kv (grouped-query attention). All
 *     tensors contiguous, on one CUDA device. Scores run in f32.
 *
 * The launch is asynchronous on `stream`; only launch errors are reported.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_paged_attention_decode(const ayaka_tensor_view_t* out,
                             const ayaka_tensor_view_t* query,
                             const ayaka_tensor_view_t* kv_cache,
                             const ayaka_tensor_view_t* block_tables,
                             const ayaka_tensor_view_t* seq_lens, float scale,
                             int32_t layout, ayaka_stream_t stream);

/*
 * Split-KV (v2) decode for long contexts: the same result as the v1 entry, but
 * the per-sequence context is partitioned and recombined with a log-sum-exp,
 * keeping more blocks resident. The caller supplies scratch sized to
 * max_partitions = ceil(max_seq_len / ayaka_paged_attention_partition_size()):
 *   - exp_sums, max_logits: F32 [num_seqs, num_heads, max_partitions]
 *   - tmp_out:              query dtype [num_seqs, num_heads, max_partitions,
 * head_dim] Other arguments match ayaka_paged_attention_decode. Async on
 * `stream`.
 */
AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_paged_attention_decode_v2(const ayaka_tensor_view_t* out,
                                const ayaka_tensor_view_t* query,
                                const ayaka_tensor_view_t* kv_cache,
                                const ayaka_tensor_view_t* block_tables,
                                const ayaka_tensor_view_t* seq_lens,
                                const ayaka_tensor_view_t* exp_sums,
                                const ayaka_tensor_view_t* max_logits,
                                const ayaka_tensor_view_t* tmp_out, float scale,
                                int32_t layout, ayaka_stream_t stream);

/* Tokens per split-KV partition (for scratch sizing). */
AYAKA_EXTERN_C AYAKA_API int32_t ayaka_paged_attention_partition_size(void);

#ifdef __cplusplus
}
#endif