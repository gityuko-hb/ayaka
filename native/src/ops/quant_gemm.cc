#include "ayaka/ops/quant_gemm.h"

#include <stdint.h>

#include "ayaka/check.h"
#include "gemm/w4a16_gemm.h"
#include "ops/common.h"

namespace {

ayaka_status_t validate_quant_gemm(const ayaka_tensor_view_t* out,
                                   const ayaka_tensor_view_t* a,
                                   const ayaka_tensor_view_t* b_quant,
                                   const ayaka_tensor_view_t* b_scales,
                                   const ayaka_tensor_view_t* b_mins,
                                   const ayaka_tensor_view_t* bias,
                                   int32_t group_size) {
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(out));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(a));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(b_quant));
  AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(b_scales));
  if (b_mins != NULL) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(b_mins));
  }
  if (bias != NULL) {
    AYAKA_RETURN_IF_ERROR(ayaka_tensor_view_validate(bias));
  }

  // All must be CUDA tensors on the same device.
  AYAKA_CHECK(out->device_kind == AYAKA_DEVICE_CUDA &&
                  a->device_kind == AYAKA_DEVICE_CUDA &&
                  b_quant->device_kind == AYAKA_DEVICE_CUDA &&
                  b_scales->device_kind == AYAKA_DEVICE_CUDA,
              "quant_gemm: out, a, b_quant, b_scales must be CUDA tensors");
  AYAKA_CHECK(out->device_ordinal == a->device_ordinal &&
                  b_quant->device_ordinal == a->device_ordinal &&
                  b_scales->device_ordinal == a->device_ordinal,
              "quant_gemm: all tensors must be on the same device");
  if (b_mins != NULL) {
    AYAKA_CHECK(b_mins->device_kind == AYAKA_DEVICE_CUDA &&
                    b_mins->device_ordinal == a->device_ordinal,
                "quant_gemm: b_mins must be a CUDA tensor on the same device");
  }
  if (bias != NULL) {
    AYAKA_CHECK(bias->device_kind == AYAKA_DEVICE_CUDA &&
                    bias->device_ordinal == a->device_ordinal,
                "quant_gemm: bias must be a CUDA tensor on the same device");
  }

  // Output and activations must be F16 or BF16.
  if (!ayaka::is_float_kernel_dtype(out->dtype)) {
    return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                             "quant_gemm: out dtype must be f16 or bf16");
  }
  if (a->dtype != out->dtype) {
    return ayaka_status_make(AYAKA_STATUS_DTYPE_ERROR,
                             "quant_gemm: a dtype must match out");
  }
  // b_quant must be U8.
  AYAKA_CHECK_EQ(b_quant->dtype, AYAKA_DTYPE_U8,
                 "quant_gemm: b_quant dtype must be U8");
  // b_scales and b_mins must match out dtype.
  AYAKA_CHECK_EQ(b_scales->dtype, out->dtype,
                 "quant_gemm: b_scales dtype must match out");
  if (b_mins != NULL) {
    AYAKA_CHECK_EQ(b_mins->dtype, out->dtype,
                   "quant_gemm: b_mins dtype must match out");
  }
  if (bias != NULL) {
    AYAKA_CHECK_EQ(bias->dtype, out->dtype,
                   "quant_gemm: bias dtype must match out");
  }

  // Rank checks.
  AYAKA_CHECK_EQ(out->rank, 2, "quant_gemm: out rank must be 2");
  AYAKA_CHECK_EQ(a->rank, 2, "quant_gemm: a rank must be 2");
  AYAKA_CHECK_EQ(b_quant->rank, 2, "quant_gemm: b_quant rank must be 2");
  AYAKA_CHECK_EQ(b_scales->rank, 2, "quant_gemm: b_scales rank must be 2");

  const int64_t M = a->shape[0];
  const int64_t K = a->shape[1];
  const int64_t N = out->shape[1];

  // Shape checks.
  AYAKA_CHECK_EQ(out->shape[0], M, "quant_gemm: out rows must equal M");
  AYAKA_CHECK_EQ(b_quant->shape[0], N, "quant_gemm: b_quant rows must equal N");
  AYAKA_CHECK_EQ(b_quant->shape[1], K / 2,
                 "quant_gemm: b_quant cols must equal K/2");
  AYAKA_CHECK_EQ(b_scales->shape[0], N,
                 "quant_gemm: b_scales rows must equal N");

  // K must be a packed-friendly size: even (two 4-bit weights per U8 byte)
  // and divisible by group_size (one scale per group).
  AYAKA_CHECK_GE(K, 1, "quant_gemm: inner dimension K must be >= 1");
  AYAKA_CHECK_EQ(K % 2, 0, "quant_gemm: K must be even");
  // group_size validation.
  AYAKA_CHECK_GT(group_size, 0, "quant_gemm: group_size must be positive");
  if (K % group_size != 0) {
    return ayaka_status_make(AYAKA_STATUS_INVALID_ARGUMENT,
                             "quant_gemm: K must be divisible by group_size");
  }
  AYAKA_CHECK_EQ(b_scales->shape[1], K / group_size,
                 "quant_gemm: b_scales cols must equal K/group_size");
  if (b_mins != NULL) {
    AYAKA_CHECK_EQ(b_mins->rank, 2, "quant_gemm: b_mins rank must be 2");
    AYAKA_CHECK_EQ(b_mins->shape[0], N, "quant_gemm: b_mins rows must equal N");
    AYAKA_CHECK_EQ(b_mins->shape[1], K / group_size,
                   "quant_gemm: b_mins cols must equal K/group_size");
  }
  if (bias != NULL) {
    AYAKA_CHECK_EQ(bias->rank, 1, "quant_gemm: bias rank must be 1");
    AYAKA_CHECK_EQ(bias->shape[0], N, "quant_gemm: bias length must equal N");
  }

  // Contiguity.
  AYAKA_CHECK(ayaka::is_row_major_contiguous(out) &&
                  ayaka::is_row_major_contiguous(a) &&
                  ayaka::is_row_major_contiguous(b_quant) &&
                  ayaka::is_row_major_contiguous(b_scales),
              "quant_gemm: all tensors must be contiguous");
  if (b_mins != NULL) {
    AYAKA_CHECK(ayaka::is_row_major_contiguous(b_mins),
                "quant_gemm: b_mins must be contiguous");
  }
  if (bias != NULL) {
    AYAKA_CHECK(ayaka::is_row_major_contiguous(bias),
                "quant_gemm: bias must be contiguous");
  }

  return ayaka_status_ok();
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t inline ayaka_quant_gemm(
    const ayaka_tensor_view_t* out, const ayaka_tensor_view_t* a,
    const ayaka_tensor_view_t* b_quant, const ayaka_tensor_view_t* b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* workspace, size_t workspace_bytes,
    ayaka_stream_t stream) {
  AYAKA_RETURN_IF_ERROR(
      validate_quant_gemm(out, a, b_quant, b_scales, b_mins, bias, group_size));
  AYAKA_CHECK(
      workspace != NULL || workspace_bytes == 0,
      "quant_gemm: non-zero workspace_bytes requires a workspace pointer");
  if (ayaka::view_numel(out) == 0) {
    return ayaka_status_ok();
  }
  return ayaka::w4a16_gemm_naive(*out, *a, *b_quant, *b_scales, b_mins, bias,
                                 group_size, workspace, workspace_bytes,
                                 stream);
}