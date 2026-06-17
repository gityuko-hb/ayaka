#pragma once

// Internal launcher contract between src/ops/paged_attention.cc and
// csrc/attention/paged_attention.cu. Not installed; no CUDA types.

#include <stdint.h>

#include "ayaka/status.h"
#include "ayaka/stream.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

// Views are pre-validated by the caller (see ayaka_paged_attention_decode).
// Returns AYAKA_STATUS_UNSUPPORTED for layouts other than NHD.
AYAKA_NODISCARD ayaka_status_t paged_attention_decode_cuda(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, float scale, int32_t layout,
    ayaka_stream_t stream);

// Split-KV (v2) decode using caller-provided scratch. `exp_sums`/`max_logits`
// are f32 [num_seqs, num_heads, max_partitions]; `tmp_out` matches query dtype
// and is [num_seqs, num_heads, max_partitions, head_dim].
AYAKA_NODISCARD ayaka_status_t paged_attention_v2_cuda(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, const ayaka_tensor_view_t& exp_sums,
    const ayaka_tensor_view_t& max_logits, const ayaka_tensor_view_t& tmp_out,
    float scale, int32_t layout, ayaka_stream_t stream);

// The split-KV partition size (tokens per partition).
int paged_attention_partition_size();

}  // namespace ayaka