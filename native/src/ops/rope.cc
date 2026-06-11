#include "ayaka/ops/rope.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "ops/common.h"
#include "pos/rope.h"

namespace {

ayaka_status_t validate_rope(const ayaka_tensor_view_t* query,
                             const ayaka_tensor_view_t* key,
                             const ayaka_tensor_view_t* positions,
                             const ayaka_tensor_view_t* cos_sin_cache,
                             int32_t layout) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(query));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(key));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(positions));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(cos_sin_cache));

    const ayaka_tensor_view_t* views[] = {query, key, positions, cos_sin_cache};
    for (const ayaka_tensor_view_t* v : views) {
        AYAKA_CHECK(v->device_kind == AYAKA_DEVICE_CUDA,
                    "rope: all tensors must be CUDA tensors");
        AYAKA_CHECK(v->device_ordinal == query->device_ordinal,
                    "rope: all tensors must be on the same device");
        AYAKA_CHECK(ayaka::is_row_major_contiguous(v),
                    "rope: all tensors must be contiguous");
    }

    if (!ayaka::is_float_kernel_dtype(query->dtype)) {
        return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                                 "rope: query dtype must be f32, f16, or bf16");
    }
    if (key->dtype != query->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "rope: key dtype must match query dtype");
    }
    if (positions->dtype != AYAKA_DTYPE_I64) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "rope: positions dtype must be i64");
    }
    if (cos_sin_cache->dtype != AYAKA_DTYPE_F32) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "rope: cos_sin_cache dtype must be f32");
    }

    AYAKA_CHECK_EQ(query->rank, 3,
                   "rope: query must be [num_tokens, num_heads, head_dim]");
    AYAKA_CHECK_EQ(key->rank, 3,
                   "rope: key must be [num_tokens, num_kv_heads, head_dim]");
    AYAKA_CHECK_EQ(positions->rank, 1, "rope: positions must be rank 1");
    AYAKA_CHECK_EQ(cos_sin_cache->rank, 2,
                   "rope: cos_sin_cache must be [max_position, rot_dim]");

    AYAKA_CHECK_EQ(key->shape[0], query->shape[0],
                   "rope: key num_tokens must match query");
    AYAKA_CHECK_EQ(positions->shape[0], query->shape[0],
                   "rope: positions length must match num_tokens");
    AYAKA_CHECK_EQ(key->shape[2], query->shape[2],
                   "rope: key head_dim must match query");

    const int64_t rot_dim = cos_sin_cache->shape[1];
    AYAKA_CHECK(rot_dim % 2 == 0, "rope: rot_dim must be even");
    AYAKA_CHECK_LE(rot_dim, query->shape[2],
                   "rope: rot_dim must not exceed head_dim");

    AYAKA_CHECK(layout == AYAKA_ROPE_LAYOUT_NEOX ||
                    layout == AYAKA_ROPE_LAYOUT_GPTJ,
                "rope: layout must be neox (0) or gptj (1)");

    AYAKA_CHECK_LE(query->shape[0], INT32_MAX,
                   "rope: token count exceeds CUDA grid limit");
    AYAKA_CHECK_LE(query->shape[1] * (rot_dim / 2), INT32_MAX,
                   "rope: heads * rot_dim/2 exceeds kernel index range");
    return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_rope(const ayaka_tensor_view_t* query,
           const ayaka_tensor_view_t* key,
           const ayaka_tensor_view_t* positions,
           const ayaka_tensor_view_t* cos_sin_cache,
           int32_t layout,
           ayaka_stream_t stream) {
    AYAKA_RETURN_IF_ERROR(
        validate_rope(query, key, positions, cos_sin_cache, layout));
    if (ayaka::view_numel(query) == 0) {
        return ayaka_status_ok();
    }
    return ayaka::rope_cuda(*query, *key, *positions, *cos_sin_cache, layout,
                            stream);
}
