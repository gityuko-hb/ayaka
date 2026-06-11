#include "ayaka/ops/rmsnorm.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "norm/rmsnorm.h"
#include "ops/common.h"

namespace {

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

    if (!ayaka::is_float_kernel_dtype(input->dtype)) {
        return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                                 "rmsnorm: dtype must be f32, f16, or bf16");
    }
    if (out->dtype != input->dtype || weight->dtype != input->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "rmsnorm: out, input, and weight dtype must match");
    }

    AYAKA_CHECK(input->rank >= 1, "rmsnorm: input rank must be >= 1");
    AYAKA_CHECK_EQ(weight->rank, 1, "rmsnorm: weight rank must be 1");
    AYAKA_CHECK(ayaka::same_shape(out, input),
                "rmsnorm: out shape must match input shape");
    AYAKA_CHECK_EQ(weight->shape[0], input->shape[input->rank - 1],
                   "rmsnorm: weight length must match input last dim");

    AYAKA_CHECK(ayaka::is_row_major_contiguous(out) &&
                    ayaka::is_row_major_contiguous(input) &&
                    ayaka::is_row_major_contiguous(weight),
                "rmsnorm: out, input, and weight must be contiguous");

    const int64_t hidden = weight->shape[0];
    if (hidden > 0) {
        AYAKA_CHECK_LE(ayaka::view_numel(input) / hidden, INT32_MAX,
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
    if (ayaka::view_numel(input) == 0) {
        return ayaka_status_ok();
    }
    return ayaka::rmsnorm_cuda(*out, *input, *weight, eps, stream);
}
