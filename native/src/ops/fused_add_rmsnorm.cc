#include "ayaka/ops/fused_add_rmsnorm.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "norm/fused_add_rmsnorm.h"
#include "ops/common.h"

namespace {

ayaka_status_t validate_fused_add_rmsnorm(const ayaka_tensor_view_t* out,
                                          const ayaka_tensor_view_t* residual_out,
                                          const ayaka_tensor_view_t* input,
                                          const ayaka_tensor_view_t* residual,
                                          const ayaka_tensor_view_t* weight) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(residual_out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(input));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(residual));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(weight));

    const ayaka_tensor_view_t* views[] = {out, residual_out, residual, weight};
    for (const ayaka_tensor_view_t* v : views) {
        AYAKA_CHECK(v->device_kind == AYAKA_DEVICE_CUDA &&
                        input->device_kind == AYAKA_DEVICE_CUDA,
                    "fused_add_rmsnorm: all tensors must be CUDA tensors");
        AYAKA_CHECK(v->device_ordinal == input->device_ordinal,
                    "fused_add_rmsnorm: all tensors must be on the same device");
        if (v->dtype != input->dtype) {
            return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                     "fused_add_rmsnorm: dtypes must match");
        }
        AYAKA_CHECK(ayaka::is_row_major_contiguous(v) &&
                        ayaka::is_row_major_contiguous(input),
                    "fused_add_rmsnorm: all tensors must be contiguous");
    }
    if (!ayaka::is_float_kernel_dtype(input->dtype)) {
        return ayaka_status_make(
            AYAKA_STATUS_UNSUPPORTED,
            "fused_add_rmsnorm: dtype must be f32, f16, or bf16");
    }

    AYAKA_CHECK(input->rank >= 1, "fused_add_rmsnorm: input rank must be >= 1");
    AYAKA_CHECK_EQ(weight->rank, 1, "fused_add_rmsnorm: weight rank must be 1");
    AYAKA_CHECK(ayaka::same_shape(out, input) &&
                    ayaka::same_shape(residual_out, input) &&
                    ayaka::same_shape(residual, input),
                "fused_add_rmsnorm: out, residual_out, and residual must match "
                "input shape");
    AYAKA_CHECK_EQ(weight->shape[0], input->shape[input->rank - 1],
                   "fused_add_rmsnorm: weight length must match input last dim");
    AYAKA_CHECK(out->data != residual_out->data,
                "fused_add_rmsnorm: out must not alias residual_out");

    const int64_t hidden = weight->shape[0];
    if (hidden > 0) {
        AYAKA_CHECK_LE(ayaka::view_numel(input) / hidden, INT32_MAX,
                       "fused_add_rmsnorm: row count exceeds CUDA grid limit");
    }
    return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_fused_add_rmsnorm(const ayaka_tensor_view_t* out,
                        const ayaka_tensor_view_t* residual_out,
                        const ayaka_tensor_view_t* input,
                        const ayaka_tensor_view_t* residual,
                        const ayaka_tensor_view_t* weight,
                        float eps,
                        ayaka_stream_t stream) {
    AYAKA_RETURN_IF_ERROR(
        validate_fused_add_rmsnorm(out, residual_out, input, residual, weight));
    AYAKA_CHECK(eps >= 0.0f, "fused_add_rmsnorm: eps must be non-negative");
    if (ayaka::view_numel(input) == 0) {
        return ayaka_status_ok();
    }
    return ayaka::fused_add_rmsnorm_cuda(*out, *residual_out, *input, *residual,
                                         *weight, eps, stream);
}
