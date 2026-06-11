#include "ayaka/ops/silu_and_mul.h"

#include <stdint.h>

#include "activation/swiglu.h"
#include "ayaka/check.h"
#include "ops/common.h"

namespace {

ayaka_status_t validate_silu_and_mul(const ayaka_tensor_view_t* out,
                                     const ayaka_tensor_view_t* input) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(input));

    AYAKA_CHECK(out->device_kind == AYAKA_DEVICE_CUDA &&
                    input->device_kind == AYAKA_DEVICE_CUDA,
                "silu_and_mul: out and input must be CUDA tensors");
    AYAKA_CHECK(out->device_ordinal == input->device_ordinal,
                "silu_and_mul: out and input must be on the same device");

    if (!ayaka::is_float_kernel_dtype(input->dtype)) {
        return ayaka_status_make(
            AYAKA_STATUS_UNSUPPORTED,
            "silu_and_mul: dtype must be f32, f16, or bf16");
    }
    if (out->dtype != input->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "silu_and_mul: out and input dtype must match");
    }

    AYAKA_CHECK(input->rank >= 1, "silu_and_mul: input rank must be >= 1");
    AYAKA_CHECK_EQ(out->rank, input->rank,
                   "silu_and_mul: out rank must match input rank");
    for (int32_t i = 0; i + 1 < input->rank; ++i) {
        AYAKA_CHECK_EQ(out->shape[i], input->shape[i],
                       "silu_and_mul: out leading dims must match input");
    }
    AYAKA_CHECK_EQ(input->shape[input->rank - 1],
                   2 * out->shape[out->rank - 1],
                   "silu_and_mul: input last dim must be 2 * out last dim");

    AYAKA_CHECK(ayaka::is_row_major_contiguous(out) &&
                    ayaka::is_row_major_contiguous(input),
                "silu_and_mul: out and input must be contiguous");
    return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_silu_and_mul(const ayaka_tensor_view_t* out,
                   const ayaka_tensor_view_t* input,
                   ayaka_stream_t stream) {
    AYAKA_RETURN_IF_ERROR(validate_silu_and_mul(out, input));
    if (ayaka::view_numel(out) == 0) {
        return ayaka_status_ok();
    }
    return ayaka::silu_and_mul_cuda(*out, *input, stream);
}
