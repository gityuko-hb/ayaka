@ @-0, 0 + 1,
    166 @ @
#include "ayaka/ops/paged_attention.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "attention/paged_attention.h"
#include "ops/common.h"

    namespace {

  ayaka_status_t validate_paged_attention_decode(
      const ayaka_tensor_view_t* out, const ayaka_tensor_view_t* query,
      const ayaka_tensor_view_t* kv_cache,
      const ayaka_tensor_view_t* block_tables,
      const ayaka_tensor_view_t* seq_lens) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(query));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(kv_cache));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(block_tables));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(seq_lens));

    const ayaka_tensor_view_t* views[] = {out, query, kv_cache, block_tables,
                                          seq_lens};
    for (const ayaka_tensor_view_t* v : views) {
      AYAKA_CHECK(v->device_kind == AYAKA_DEVICE_CUDA,
                  "paged_attention_decode: all tensors must be CUDA tensors");
      AYAKA_CHECK(
          v->device_ordinal == query->device_ordinal,
          "paged_attention_decode: all tensors must be on the same device");
      AYAKA_CHECK(ayaka::is_row_major_contiguous(v),
                  "paged_attention_decode: all tensors must be contiguous");
    }

    if (!ayaka::is_float_kernel_dtype(query->dtype)) {
      return ayaka_status_make(
          AYAKA_STATUS_UNSUPPORTED,
          "paged_attention_decode: query dtype must be f32, f16, or bf16");
    }
    if (out->dtype != query->dtype) {
      return ayaka_status_make(
          AYAKA_STATUS_DTYPE_ERROR,
          "paged_attention_decode: out dtype must match query dtype");
    }
    if (kv_cache->dtype != query->dtype) {
      return ayaka_status_make(
          AYAKA_STATUS_DTYPE_ERROR,
          "paged_attention_decode: kv_cache dtype must match query dtype");
    }
    if (block_tables->dtype != AYAKA_DTYPE_I64) {
      return ayaka_status_make(
          AYAKA_STATUS_DTYPE_ERROR,
          "paged_attention_decode: block_tables dtype must be i64");
    }
    if (seq_lens->dtype != AYAKA_DTYPE_I64) {
      return ayaka_status_make(
          AYAKA_STATUS_DTYPE_ERROR,
          "paged_attention_decode: seq_lens dtype must be i64");
    }

    AYAKA_CHECK_EQ(out->rank, 3,
                   "paged_attention_decode: out must be [S, H, D]");
    AYAKA_CHECK_EQ(query->rank, 3,
                   "paged_attention_decode: query must be [S, H, D]");
    AYAKA_CHECK_EQ(kv_cache->rank, 5,
                   "paged_attention_decode: kv_cache must be "
                   "[num_blocks, 2, block_size, H_kv, D]");
    AYAKA_CHECK_EQ(
        block_tables->rank, 2,
        "paged_attention_decode: block_tables must be [S, max_blocks]");
    AYAKA_CHECK_EQ(seq_lens->rank, 1,
                   "paged_attention_decode: seq_lens must be [S]");

    AYAKA_CHECK(ayaka::same_shape(out, query),
                "paged_attention_decode: out shape must match query");
    const int64_t num_seqs = query->shape[0];
    const int64_t num_heads = query->shape[1];
    const int64_t head_dim = query->shape[2];
    AYAKA_CHECK_EQ(
        kv_cache->shape[1], 2,
        "paged_attention_decode: kv_cache dim 1 must be 2 (K and V)");
    const int64_t h_kv = kv_cache->shape[3];
    AYAKA_CHECK_EQ(kv_cache->shape[4], head_dim,
                   "paged_attention_decode: kv_cache D must match query D");
    AYAKA_CHECK(num_heads % h_kv == 0,
                "paged_attention_decode: num_heads must be a multiple of H_kv");
    AYAKA_CHECK_EQ(block_tables->shape[0], num_seqs,
                   "paged_attention_decode: block_tables rows must equal S");
    AYAKA_CHECK_EQ(seq_lens->shape[0], num_seqs,
                   "paged_attention_decode: seq_lens length must equal S");

    return ayaka_status_ok();
  }

}  // namespace

extern "C" AYAKA_API ayaka_status_t ayaka_paged_attention_decode(
    const ayaka_tensor_view_t* out, const ayaka_tensor_view_t* query,
    const ayaka_tensor_view_t* kv_cache,
    const ayaka_tensor_view_t* block_tables,
    const ayaka_tensor_view_t* seq_lens, float scale, int32_t layout,
    ayaka_stream_t stream) {
  AYAKA_RETURN_IF_ERROR(validate_paged_attention_decode(
      out, query, kv_cache, block_tables, seq_lens));
  if (ayaka::view_numel(query) == 0) {
    return ayaka_status_ok();
  }
  return ayaka::paged_attention_decode_cuda(
      *out, *query, *kv_cache, *block_tables, *seq_lens, scale, layout, stream);
}

extern "C" AYAKA_API ayaka_status_t ayaka_paged_attention_decode_v2(
    const ayaka_tensor_view_t* out, const ayaka_tensor_view_t* query,
    const ayaka_tensor_view_t* kv_cache,
    const ayaka_tensor_view_t* block_tables,
    const ayaka_tensor_view_t* seq_lens, const ayaka_tensor_view_t* exp_sums,
    const ayaka_tensor_view_t* max_logits, const ayaka_tensor_view_t* tmp_out,
    float scale, int32_t layout, ayaka_stream_t stream) {
  AYAKA_RETURN_IF_ERROR(validate_paged_attention_decode(
      out, query, kv_cache, block_tables, seq_lens));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(exp_sums));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(max_logits));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(tmp_out));
  const ayaka_tensor_view_t* scratch[] = {exp_sums, max_logits, tmp_out};
  for (const ayaka_tensor_view_t* v : scratch) {
    AYAKA_CHECK(v->device_kind == AYAKA_DEVICE_CUDA,
                "paged_attention_decode_v2: scratch must be CUDA tensors");
    AYAKA_CHECK(v->device_ordinal == query->device_ordinal,
                "paged_attention_decode_v2: scratch must be on query's device");
    AYAKA_CHECK(ayaka::is_row_major_contiguous(v),
                "paged_attention_decode_v2: scratch must be contiguous");
  }
  AYAKA_CHECK_EQ(exp_sums->dtype, AYAKA_DTYPE_F32,
                 "paged_attention_decode_v2: exp_sums dtype must be f32");
  AYAKA_CHECK_EQ(max_logits->dtype, AYAKA_DTYPE_F32,
                 "paged_attention_decode_v2: max_logits dtype must be f32");
  AYAKA_CHECK_EQ(tmp_out->dtype, query->dtype,
                 "paged_attention_decode_v2: tmp_out dtype must match query");
  const int64_t num_seqs = query->shape[0];
  const int64_t num_heads = query->shape[1];
  const int64_t head_dim = query->shape[2];
  AYAKA_CHECK_EQ(exp_sums->rank, 3,
                 "paged_attention_decode_v2: exp_sums must be [S, H, P]");
  AYAKA_CHECK_EQ(max_logits->rank, 3,
                 "paged_attention_decode_v2: max_logits must be [S, H, P]");
  AYAKA_CHECK_EQ(tmp_out->rank, 4,
                 "paged_attention_decode_v2: tmp_out must be [S, H, P, D]");
  AYAKA_CHECK_EQ(exp_sums->shape[0], num_seqs,
                 "paged_attention_decode_v2: exp_sums S must match query");
  AYAKA_CHECK_EQ(exp_sums->shape[1], num_heads,
                 "paged_attention_decode_v2: exp_sums H must match query");
  AYAKA_CHECK(
      ayaka::same_shape(exp_sums, max_logits),
      "paged_attention_decode_v2: exp_sums and max_logits shape must match");
  const int64_t max_partitions = exp_sums->shape[2];
  AYAKA_CHECK_EQ(tmp_out->shape[0], num_seqs,
                 "paged_attention_decode_v2: tmp_out S must match query");
  AYAKA_CHECK_EQ(tmp_out->shape[1], num_heads,
                 "paged_attention_decode_v2: tmp_out H must match query");
  AYAKA_CHECK_EQ(tmp_out->shape[2], max_partitions,
                 "paged_attention_decode_v2: tmp_out P must match exp_sums");
  AYAKA_CHECK_EQ(tmp_out->shape[3], head_dim,
                 "paged_attention_decode_v2: tmp_out D must match query");
  if (ayaka::view_numel(query) == 0) {
    return ayaka_status_ok();
  }
  return ayaka::paged_attention_v2_cuda(*out, *query, *kv_cache, *block_tables,
                                        *seq_lens, *exp_sums, *max_logits,
                                        *tmp_out, scale, layout, stream);
}

extern "C" AYAKA_API int32_t ayaka_paged_attention_partition_size(void) {
  return static_cast<int32_t>(ayaka::paged_attention_partition_size());
}
