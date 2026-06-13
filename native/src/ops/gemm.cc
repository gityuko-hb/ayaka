#include "ayaka/ops/gemm.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "backends/cublaslt_gemm.h"
#include "ops/common.h"

namespace {

ayaka_status_t validate_gemm(const ayaka_tensor_view_t* out,
                             const ayaka_tensor_view_t* a,
                             const ayaka_tensor_view_t* b,
                             const ayaka_tensor_view_t* bias,
                             int32_t trans_a,
                             int32_t trans_b) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(a));
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(b));
    if (bias != NULL) {
        AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(bias));
    }

    AYAKA_CHECK(out->device_kind == AYAKA_DEVICE_CUDA &&
                    a->device_kind == AYAKA_DEVICE_CUDA &&
                    b->device_kind == AYAKA_DEVICE_CUDA,
                "gemm: out, a, and b must be CUDA tensors");
    AYAKA_CHECK(out->device_ordinal == a->device_ordinal &&
                    b->device_ordinal == a->device_ordinal,
                "gemm: out, a, and b must be on the same device");
    if (bias != NULL) {
        AYAKA_CHECK(bias->device_kind == AYAKA_DEVICE_CUDA &&
                        bias->device_ordinal == a->device_ordinal,
                    "gemm: bias must be a CUDA tensor on a's device");
    }

    if (!ayaka::is_float_kernel_dtype(out->dtype)) {
        return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                                 "gemm: dtype must be f32, f16, or bf16");
    }
    if (a->dtype != out->dtype || b->dtype != out->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "gemm: out, a, and b dtype must match");
    }
    if (bias != NULL && bias->dtype != out->dtype) {
        return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                                 "gemm: bias dtype must match out");
    }

    AYAKA_CHECK_EQ(out->rank, 2, "gemm: out rank must be 2");
    AYAKA_CHECK_EQ(a->rank, 2, "gemm: a rank must be 2");
    AYAKA_CHECK_EQ(b->rank, 2, "gemm: b rank must be 2");

    const int64_t m = trans_a ? a->shape[1] : a->shape[0];
    const int64_t k = trans_a ? a->shape[0] : a->shape[1];
    const int64_t k_b = trans_b ? b->shape[1] : b->shape[0];
    const int64_t n = trans_b ? b->shape[0] : b->shape[1];
    AYAKA_CHECK_EQ(k, k_b,
                   "gemm: inner dims of op_a(a) and op_b(b) must match");
    AYAKA_CHECK_EQ(out->shape[0], m, "gemm: out rows must equal M");
    AYAKA_CHECK_EQ(out->shape[1], n, "gemm: out cols must equal N");
    if (bias != NULL) {
        AYAKA_CHECK_EQ(bias->rank, 1, "gemm: bias rank must be 1");
        AYAKA_CHECK_EQ(bias->shape[0], n, "gemm: bias length must equal N");
    }

    AYAKA_CHECK(ayaka::is_row_major_contiguous(out) &&
                    ayaka::is_row_major_contiguous(a) &&
                    ayaka::is_row_major_contiguous(b),
                "gemm: out, a, and b must be contiguous");
    if (bias != NULL) {
        AYAKA_CHECK(ayaka::is_row_major_contiguous(bias),
                    "gemm: bias must be contiguous");
    }
    return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_gemm(const ayaka_tensor_view_t* out,
           const ayaka_tensor_view_t* a,
           const ayaka_tensor_view_t* b,
           const ayaka_tensor_view_t* bias,
           int32_t trans_a,
           int32_t trans_b,
           void* workspace,
           size_t workspace_bytes,
           ayaka_stream_t stream) {
    AYAKA_RETURN_IF_ERROR(validate_gemm(out, a, b, bias, trans_a, trans_b));
    AYAKA_CHECK(workspace != NULL || workspace_bytes == 0,
                "gemm: non-zero workspace_bytes requires a workspace pointer");
    if (ayaka::view_numel(out) == 0) {
        return ayaka_status_ok();
    }
    const int64_t k = trans_a ? a->shape[0] : a->shape[1];
    AYAKA_CHECK_GE(k, 1, "gemm: inner dimension K must be >= 1");
    return ayaka::gemm_cublaslt(*out, *a, *b, bias, trans_a != 0,
                                trans_b != 0, workspace, workspace_bytes,
                                stream);
}
