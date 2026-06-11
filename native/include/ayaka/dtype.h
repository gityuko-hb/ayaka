#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AYAKA_DTYPE_F32     = 0,
    AYAKA_DTYPE_F16     = 1,
    AYAKA_DTYPE_BF16    = 2,
    AYAKA_DTYPE_F8E4M3  = 3,
    AYAKA_DTYPE_F8E5M2  = 4,
    AYAKA_DTYPE_F4E2M1  = 5,
    AYAKA_DTYPE_F8E8M0  = 6,
    AYAKA_DTYPE_I64     = 7,
    AYAKA_DTYPE_I32     = 8,
    AYAKA_DTYPE_I16     = 9,
    AYAKA_DTYPE_I8      = 10,
    AYAKA_DTYPE_U8      = 11,
    AYAKA_DTYPE_I4      = 12,
    AYAKA_DTYPE_U4      = 13,
    AYAKA_DTYPE_BOOL    = 14,
} ayaka_dtype_t;

#ifdef __cplusplus
}
#endif
