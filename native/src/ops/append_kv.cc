#include "ayaka/ops/append_kv.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "kv_cache/append_kv.h"
#include "ops/common.h"

namespace {

ayaka_status_t validate_append_kv(const ayaka_tensor_view_t* key,
                                  const ayaka_tensor_view_t* value,
                                  const ayaka_tensor_view_t* kv_cache,
                                  const ayaka_tensor_view_t* slot_mapping,
                                  int32_t layout) {
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(key));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(value));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(kv_cache));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(slot_mapping));

  const ayaka_tensor_view_t* views[] = {key, value, kv_cache, slot_mapping};
  for (const ayaka_tensor_view_t* v : views) {
    AYAKA_CHECK(v->device_kind == AYAKA_DEVICE_CUDA,
                "append_kv: all tensors must be CUDA tensors");
    AYAKA_CHECK(v->device_ordinal == key->device_ordinal,
                "append_kv: all tensors must be on the same device");
    AYAKA_CHECK(ayaka::is_row_major_contiguous(v),
                "append_kv: all tensors must be contiguous");
  }

  if (!ayaka::is_float_kernel_dtype(key->dtype)) {
    return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                             "append_kv: key dtype must be f32, f16, or bf16");
  }
  if (value->dtype != key->dtype) {
    return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                             "append_kv: value dtype must match key dtype");
  }
  if (kv_cache->dtype != key->dtype) {
    return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                             "append_kv: kv_cache dtype must match key dtype");
  }
  if (slot_mapping->dtype != AYAKA_DTYPE_I64) {
    return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                             "append_kv: slot_mapping dtype must be i64");
  }

  AYAKA_CHECK_EQ(key->rank, 3, "append_kv: key must be [N, H_kv, D]");
  AYAKA_CHECK_EQ(value->rank, 3, "append_kv: value must be [N, H_kv, D]");
  AYAKA_CHECK_EQ(kv_cache->rank, 5,
                 "append_kv: kv_cache must be "
                 "[num_blocks, 2, block_size, H_kv, D]");
  AYAKA_CHECK_EQ(slot_mapping->rank, 1,
                 "append_kv: slot_mapping must be rank 1");

  AYAKA_CHECK(ayaka::same_shape(value, key),
              "append_kv: value shape must match key");
  AYAKA_CHECK_EQ(kv_cache->shape[1], 2,
                 "append_kv: kv_cache dim 1 must be 2 (K and V)");
  AYAKA_CHECK_EQ(kv_cache->shape[3], key->shape[1],
                 "append_kv: kv_cache H_kv must match key");
  AYAKA_CHECK_EQ(kv_cache->shape[4], key->shape[2],
                 "append_kv: kv_cache D must match key");
  AYAKA_CHECK_EQ(slot_mapping->shape[0], key->shape[0],
                 "append_kv: slot_mapping length must match N");

  AYAKA_CHECK(layout == AYAKA_KV_LAYOUT_NHD || layout == AYAKA_KV_LAYOUT_HND,
              "append_kv: layout must be nhd (0) or hnd (1)");

  AYAKA_CHECK_LE(key->shape[0], INT32_MAX,
                 "append_kv: token count exceeds CUDA grid limit");
  return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t ayaka_append_kv(
    const ayaka_tensor_view_t* key, const ayaka_tensor_view_t* value,
    const ayaka_tensor_view_t* kv_cache,
    const ayaka_tensor_view_t* slot_mapping, int32_t layout,
    ayaka_stream_t stream) {
  AYAKA_RETURN_IF_ERROR(
      validate_append_kv(key, value, kv_cache, slot_mapping, layout));
  if (ayaka::view_numel(key) == 0) {
    return ayaka_status_ok();
  }
  return ayaka::append_kv_cuda(*key, *value, *kv_cache, *slot_mapping, layout,
                               stream);
}