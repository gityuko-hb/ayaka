#pragma once

#include <stdint.h>
#include "ayaka/desc_header.h"
#include "ayaka/dtype.h"
#include "ayaka/device.h"
#include "ayaka/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AYAKA_MAX_RANK 8

typedef struct ayaka_tensor_view {
    AYAKA_DESC_HEADER();
    void*    data;
    int64_t  shape[AYAKA_MAX_RANK];
    int64_t  stride[AYAKA_MAX_RANK];
    int32_t  rank;
    int32_t  dtype;
    int32_t  device_kind;
    int32_t  device_ordinal;
    uint32_t flags;
    uint32_t _reserved[3];
} ayaka_tensor_view_t;

#ifdef __cplusplus
static_assert(sizeof(ayaka_tensor_view_t) == 176,
              "ayaka_tensor_view_t must be 176 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ayaka_tensor_view_t) == 176,
               "ayaka_tensor_view_t must be 176 bytes");
#endif

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_tensor_view_validate(const ayaka_tensor_view_t* view);

#ifdef __cplusplus
}
#endif
