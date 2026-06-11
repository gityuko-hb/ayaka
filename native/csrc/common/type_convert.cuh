#pragma once

// Storage-dtype <-> f32 conversion helpers shared by all kernels.
// Accumulation is always f32; storage types are f32/f16/bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ayaka {

__device__ __forceinline__ float to_float(float v) {
    return v;
}
__device__ __forceinline__ float to_float(__half v) {
    return __half2float(v);
}
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

}  // namespace ayaka
