#pragma once

#include <stddef.h>
#include <stdint.h>

#include "ayaka/export.h"
#include "ayaka/status.h"
#include "ayaka/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AYAKA_MEMCPY_HOST_TO_DEVICE = 1,
    AYAKA_MEMCPY_DEVICE_TO_HOST = 2,
    AYAKA_MEMCPY_DEVICE_TO_DEVICE = 3,
} ayaka_memcpy_kind_t;

typedef struct {
    size_t free_bytes;
    size_t total_bytes;
} ayaka_mem_info_t;

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_device_alloc(void** out, size_t bytes, int32_t device_ordinal);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_device_free(void* ptr, int32_t device_ordinal);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_device_alloc_async(void** out,
                             size_t bytes,
                             int32_t device_ordinal,
                             ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_device_free_async(void* ptr, int32_t device_ordinal, ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_host_pinned_alloc(void** out, size_t bytes);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_host_pinned_free(void* ptr);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_memcpy_async(void* dst,
                       const void* src,
                       size_t bytes,
                       ayaka_memcpy_kind_t kind,
                       ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_memset_async(void* dst, int value, size_t bytes, ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_get_info(ayaka_mem_info_t* out, int32_t device_ordinal);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_stream_synchronize(ayaka_stream_t stream);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_pool_supported(int* out_supported, int32_t device_ordinal);

AYAKA_EXTERN_C AYAKA_API AYAKA_NODISCARD ayaka_status_t
ayaka_mem_pool_trim(size_t min_bytes_to_keep, int32_t device_ordinal);

#ifdef __cplusplus
}
#endif
