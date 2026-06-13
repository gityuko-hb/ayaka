#pragma once

// Shared host-side validation helpers for op entry points
// (native/src/ops/*.cc). Internal; not installed.

#include <stdint.h>

#include "ayaka/check.h"
#include "ayaka/tensor_view.h"

namespace ayaka {

AYAKA_NODISCARD inline int64_t view_numel(const ayaka_tensor_view_t* v) {
  int64_t n = 1;
  for (int32_t i = 0; i < v->rank; ++i) {
    n *= v->shape[i];
  }
  return n;
}

// Product of every dim except the last: the row count for ops that reduce
// or normalize over the last dimension.
AYAKA_NODISCARD inline int64_t view_outer_rows(const ayaka_tensor_view_t* v) {
  int64_t rows = 1;
  for (int32_t i = 0; i + 1 < v->rank; ++i) {
    rows *= v->shape[i];
  }
  return rows;
}

AYAKA_NODISCARD inline bool is_row_major_contiguous(
    const ayaka_tensor_view_t* v) {
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

AYAKA_NODISCARD inline bool is_float_kernel_dtype(int32_t dtype) {
  return dtype == AYAKA_DTYPE_F32 || dtype == AYAKA_DTYPE_F16 ||
         dtype == AYAKA_DTYPE_BF16;
}

AYAKA_NODISCARD inline bool same_shape(const ayaka_tensor_view_t* a,
                                       const ayaka_tensor_view_t* b) {
  if (a->rank != b->rank) {
    return false;
  }
  for (int32_t i = 0; i < a->rank; ++i) {
    if (a->shape[i] != b->shape[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace ayaka
