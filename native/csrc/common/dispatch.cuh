#pragma once

// Dtype dispatch for templated kernel launchers, in the spirit of
// PyTorch's AT_DISPATCH_* macros.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ayaka/dtype.h"
#include "ayaka/error.h"

// Invoke `...` (a zero-argument callable, typically `[&] { ... }`) with
// `scalar_t` bound to the storage type for DTYPE, returning its
// ayaka_status_t. Unsupported dtypes return AYAKA_STATUS_UNSUPPORTED with
// OP_NAME (a string literal) in the message.
#define AYAKA_DISPATCH_FLOAT_DTYPES(DTYPE, OP_NAME, ...)                 \
    switch (DTYPE) {                                                     \
        case AYAKA_DTYPE_F32: {                                          \
            using scalar_t = float;                                      \
            return (__VA_ARGS__)();                                      \
        }                                                                \
        case AYAKA_DTYPE_F16: {                                          \
            using scalar_t = __half;                                     \
            return (__VA_ARGS__)();                                      \
        }                                                                \
        case AYAKA_DTYPE_BF16: {                                         \
            using scalar_t = __nv_bfloat16;                              \
            return (__VA_ARGS__)();                                      \
        }                                                                \
        default:                                                         \
            return ayaka_status_make(                                    \
                AYAKA_STATUS_UNSUPPORTED,                                \
                OP_NAME ": dtype must be f32, f16, or bf16");            \
    }
