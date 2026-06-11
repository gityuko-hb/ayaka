#include "ayaka/memory.h"

#include <cuda_runtime.h>

#include "ayaka/check.h"
#include "guard.h"

namespace {

ayaka_status_t cuda_status(cudaError_t err, ayaka_status_code_t code) {
    if (err == cudaSuccess) {
        return ayaka_status_ok();
    }
    return ayaka_status_make(code, cudaGetErrorString(err));
}

ayaka_status_t set_device(int32_t device_ordinal) {
    AYAKA_CHECK(device_ordinal >= 0, "memory: device_ordinal must be non-negative");
    return cuda_status(cudaSetDevice(device_ordinal), AYAKA_STATUS_DEVICE_ERROR);
}

ayaka_status_t memcpy_kind(ayaka_memcpy_kind_t kind, cudaMemcpyKind* out) {
    AYAKA_CHECK_NONNULL(out, "memory: memcpy kind out pointer is null");
    switch (kind) {
        case AYAKA_MEMCPY_HOST_TO_DEVICE:
            *out = cudaMemcpyHostToDevice;
            return ayaka_status_ok();
        case AYAKA_MEMCPY_DEVICE_TO_HOST:
            *out = cudaMemcpyDeviceToHost;
            return ayaka_status_ok();
        case AYAKA_MEMCPY_DEVICE_TO_DEVICE:
            *out = cudaMemcpyDeviceToDevice;
            return ayaka_status_ok();
        default:
            return ayaka_status_make(AYAKA_STATUS_INVALID_ARGUMENT,
                                     "memory: unknown memcpy kind");
    }
}

ayaka_status_t device_alloc(void** out, size_t bytes, int32_t device_ordinal) {
    AYAKA_CHECK_NONNULL(out, "memory: device alloc out pointer is null");
    *out = nullptr;
    if (bytes == 0) {
        return ayaka_status_ok();
    }
    AYAKA_RETURN_IF_ERROR(set_device(device_ordinal));
    return cuda_status(cudaMalloc(out, bytes), AYAKA_STATUS_OUT_OF_MEMORY);
}

ayaka_status_t device_free(void* ptr, int32_t device_ordinal) {
    if (ptr == nullptr) {
        return ayaka_status_ok();
    }
    AYAKA_RETURN_IF_ERROR(set_device(device_ordinal));
    return cuda_status(cudaFree(ptr), AYAKA_STATUS_INTERNAL_ERROR);
}

ayaka_status_t device_alloc_async(void** out,
                                  size_t bytes,
                                  int32_t device_ordinal,
                                  ayaka_stream_t stream) {
    AYAKA_CHECK_NONNULL(out, "memory: async device alloc out pointer is null");
    *out = nullptr;
    if (bytes == 0) {
        return ayaka_status_ok();
    }
    AYAKA_RETURN_IF_ERROR(set_device(device_ordinal));
    return cuda_status(cudaMallocAsync(out, bytes, reinterpret_cast<cudaStream_t>(stream)),
                       AYAKA_STATUS_OUT_OF_MEMORY);
}

ayaka_status_t device_free_async(void* ptr, int32_t device_ordinal, ayaka_stream_t stream) {
    if (ptr == nullptr) {
        return ayaka_status_ok();
    }
    AYAKA_RETURN_IF_ERROR(set_device(device_ordinal));
    return cuda_status(cudaFreeAsync(ptr, reinterpret_cast<cudaStream_t>(stream)),
                       AYAKA_STATUS_INTERNAL_ERROR);
}

ayaka_status_t host_pinned_alloc(void** out, size_t bytes) {
    AYAKA_CHECK_NONNULL(out, "memory: pinned alloc out pointer is null");
    *out = nullptr;
    if (bytes == 0) {
        return ayaka_status_ok();
    }
    return cuda_status(cudaHostAlloc(out, bytes, cudaHostAllocDefault),
                       AYAKA_STATUS_OUT_OF_MEMORY);
}

ayaka_status_t host_pinned_free(void* ptr) {
    if (ptr == nullptr) {
        return ayaka_status_ok();
    }
    return cuda_status(cudaFreeHost(ptr), AYAKA_STATUS_INTERNAL_ERROR);
}

ayaka_status_t memcpy_async(void* dst,
                            const void* src,
                            size_t bytes,
                            ayaka_memcpy_kind_t kind,
                            ayaka_stream_t stream) {
    if (bytes == 0) {
        return ayaka_status_ok();
    }
    AYAKA_CHECK_NONNULL(dst, "memory: memcpy dst is null");
    AYAKA_CHECK_NONNULL(src, "memory: memcpy src is null");
    cudaMemcpyKind cuda_kind;
    AYAKA_RETURN_IF_ERROR(memcpy_kind(kind, &cuda_kind));
    return cuda_status(cudaMemcpyAsync(dst, src, bytes, cuda_kind,
                                       reinterpret_cast<cudaStream_t>(stream)),
                       AYAKA_STATUS_INTERNAL_ERROR);
}

ayaka_status_t memset_async(void* dst, int value, size_t bytes, ayaka_stream_t stream) {
    if (bytes == 0) {
        return ayaka_status_ok();
    }
    AYAKA_CHECK_NONNULL(dst, "memory: memset dst is null");
    return cuda_status(cudaMemsetAsync(dst, value, bytes,
                                       reinterpret_cast<cudaStream_t>(stream)),
                       AYAKA_STATUS_INTERNAL_ERROR);
}

ayaka_status_t get_info(ayaka_mem_info_t* out, int32_t device_ordinal) {
    AYAKA_CHECK_NONNULL(out, "memory: get_info out pointer is null");
    AYAKA_RETURN_IF_ERROR(set_device(device_ordinal));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    AYAKA_RETURN_IF_ERROR(cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes),
                                       AYAKA_STATUS_INTERNAL_ERROR));
    out->free_bytes = free_bytes;
    out->total_bytes = total_bytes;
    return ayaka_status_ok();
}

ayaka_status_t stream_synchronize(ayaka_stream_t stream) {
    return cuda_status(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                       AYAKA_STATUS_STREAM_ERROR);
}

ayaka_status_t pool_supported(int* out_supported, int32_t device_ordinal) {
    AYAKA_CHECK_NONNULL(out_supported, "memory: pool_supported out pointer is null");
    int supported = 0;
    AYAKA_RETURN_IF_ERROR(cuda_status(
        cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device_ordinal),
        AYAKA_STATUS_DEVICE_ERROR));
    *out_supported = supported != 0 ? 1 : 0;
    return ayaka_status_ok();
}

ayaka_status_t pool_trim(size_t min_bytes_to_keep, int32_t device_ordinal) {
    cudaMemPool_t pool = nullptr;
    AYAKA_RETURN_IF_ERROR(cuda_status(cudaDeviceGetDefaultMemPool(&pool, device_ordinal),
                                       AYAKA_STATUS_DEVICE_ERROR));
    return cuda_status(cudaMemPoolTrimTo(pool, min_bytes_to_keep),
                       AYAKA_STATUS_INTERNAL_ERROR);
}

}  // namespace

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_device_alloc(void** out, size_t bytes, int32_t device_ordinal) {
    AYAKA_C_API_GUARD(device_alloc(out, bytes, device_ordinal));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_device_free(void* ptr, int32_t device_ordinal) {
    AYAKA_C_API_GUARD(device_free(ptr, device_ordinal));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_device_alloc_async(void** out,
                             size_t bytes,
                             int32_t device_ordinal,
                             ayaka_stream_t stream) {
    AYAKA_C_API_GUARD(device_alloc_async(out, bytes, device_ordinal, stream));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_device_free_async(void* ptr, int32_t device_ordinal, ayaka_stream_t stream) {
    AYAKA_C_API_GUARD(device_free_async(ptr, device_ordinal, stream));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_host_pinned_alloc(void** out, size_t bytes) {
    AYAKA_C_API_GUARD(host_pinned_alloc(out, bytes));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_host_pinned_free(void* ptr) {
    AYAKA_C_API_GUARD(host_pinned_free(ptr));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_memcpy_async(void* dst,
                       const void* src,
                       size_t bytes,
                       ayaka_memcpy_kind_t kind,
                       ayaka_stream_t stream) {
    AYAKA_C_API_GUARD(memcpy_async(dst, src, bytes, kind, stream));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_memset_async(void* dst, int value, size_t bytes, ayaka_stream_t stream) {
    AYAKA_C_API_GUARD(memset_async(dst, value, bytes, stream));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_get_info(ayaka_mem_info_t* out, int32_t device_ordinal) {
    AYAKA_C_API_GUARD(get_info(out, device_ordinal));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_stream_synchronize(ayaka_stream_t stream) {
    AYAKA_C_API_GUARD(stream_synchronize(stream));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_pool_supported(int* out_supported, int32_t device_ordinal) {
    AYAKA_C_API_GUARD(pool_supported(out_supported, device_ordinal));
}

extern "C" AYAKA_API ayaka_status_t
ayaka_mem_pool_trim(size_t min_bytes_to_keep, int32_t device_ordinal) {
    AYAKA_C_API_GUARD(pool_trim(min_bytes_to_keep, device_ordinal));
}
