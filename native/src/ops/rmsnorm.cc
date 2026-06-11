#include "ayaka/ops/rmsnorm.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "norm/rmsnorm.h"

namespace {

int64_t numel(const ayaka_tensor_view_t* v) {
    int64_t n = 1;
    for (int32_t i = 0; i < v->rank; ++i) {
        n *= v->shape[i];
    }
    return n;
}

bool is_row_major_contiguous(const ayaka_tensor_view_t* v) {
    int64_t expected = 1;
    for (int32_t i = v->rank - 1; i >= 0; --i) {
        if (v->shape[i] == 0) {
            return true;  // empty tensor, layout irrelevant
        }
        if (v->shape[i] != 1 && v->stride[i] != expected) {
            return false;
        }
        expected *= v->shape[i];
    }
    return true;
}

ayaka_status_t validate_rmsnorm(const ayaka_tensor_view_t* out,
                                const ayaka_tensor_view_t* input,
                                const ayaka_tensor_view_t* weight) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(input));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(weight));

    AYAKA_CHECK(out->device_kind == AYAKA_DEVICE_CUDA &&
                    input->device_kind == AYAKA_DEVICE_CUDA &&
                    weight->device_kind == AYAKA_DEVICE_CUDA,
                "rmsnorm: out, input, and weight must be CUDA tensors");
    AYAKA_CHECK(out->device_ordinal == input->device_ordinal &&
                    weight->device_ordinal == input->device_ordinal,
                "rmsnorm: out, input, and weight must be on the same device");

    if (input->dtype != AYAKA_DTYPE_F32 && input->dtype != AYAKA_DTYPE_F16 &&
        input->dtype != AYAKA_DTYPE_BF16) {
        return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                                 "rmsnorm: dtype must be f32, f16, or bf16");
    }
    if (out->dtype != input->dtype || weight->dtype != input->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "rmsnorm: out, input, and weight dtype must match");
    }

    AYAKA_CHECK(input->rank >= 1, "rmsnorm: input rank must be >= 1");
    AYAKA_CHECK_EQ(weight->rank, 1, "rmsnorm: weight rank must be 1");
    AYAKA_CHECK_EQ(out->rank, input->rank,
                   "rmsnorm: out rank must match input rank");
    for (int32_t i = 0; i < input->rank; ++i) {
        AYAKA_CHECK_EQ(out->shape[i], input->shape[i],
                       "rmsnorm: out shape must match input shape");
    }
    AYAKA_CHECK_EQ(weight->shape[0], input->shape[input->rank - 1],
                   "rmsnorm: weight length must match input last dim");

    AYAKA_CHECK(is_row_major_contiguous(out) && is_row_major_contiguous(input) &&
                    is_row_major_contiguous(weight),
                "rmsnorm: out, input, and weight must be contiguous");

    const int64_t hidden = weight->shape[0];
    if (hidden > 0) {
        AYAKA_CHECK_LE(numel(input) / hidden, INT32_MAX,
                       "rmsnorm: row count exceeds CUDA grid limit");
    }
    return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_rmsnorm(const ayaka_tensor_view_t* out,
              const ayaka_tensor_view_t* input,
              const ayaka_tensor_view_t* weight,
              float eps,
              ayaka_stream_t stream) {
    AYAKA_RETURN_IF_ERROR(validate_rmsnorm(out, input, weight));
    AYAKA_CHECK(eps >= 0.0f, "rmsnorm: eps must be non-negative");
    if (numel(input) == 0) {
        return ayaka_status_ok();
    }
    return ayaka::rmsnorm_cuda(*out, *input, *weight, eps, stream);
}
