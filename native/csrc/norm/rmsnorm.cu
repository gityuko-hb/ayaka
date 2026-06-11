#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "norm/rmsnorm.h"

namespace ayaka {
namespace {

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

// Sum `v` across the block; every thread returns the total.
template <int BLOCK>
__device__ float block_reduce_sum(float v) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be whole warps");
    constexpr int NUM_WARPS = BLOCK / 32;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float total;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            total = v;
        }
    }
    __syncthreads();
    return total;
}

// One block per row; f32 accumulation independent of storage dtype.
template <typename T, int BLOCK>
__global__ void rmsnorm_kernel(T* __restrict__ out,
                               const T* __restrict__ input,
                               const T* __restrict__ weight,
                               int64_t hidden,
                               float eps) {
    const int64_t row = blockIdx.x;
    const T* row_in = input + row * hidden;
    T* row_out = out + row * hidden;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
        const float v = to_float(row_in[i]);
        sum_sq = fmaf(v, v, sum_sq);
    }

    const float total = block_reduce_sum<BLOCK>(sum_sq);
    const float inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);

    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
        const float v = to_float(row_in[i]) * inv_rms * to_float(weight[i]);
        row_out[i] = from_float<T>(v);
    }
}

template <typename T>
ayaka_status_t launch_rmsnorm(const ayaka_tensor_view_t& out,
                              const ayaka_tensor_view_t& input,
                              const ayaka_tensor_view_t& weight,
                              int64_t rows,
                              int64_t hidden,
                              float eps,
                              cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const dim3 grid(static_cast<unsigned int>(rows));

    rmsnorm_kernel<T, BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<T*>(out.data),
        static_cast<const T*>(input.data),
        static_cast<const T*>(weight.data),
        hidden,
        eps);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ayaka_status_make(AYAKA_STATUS_KERNEL_LAUNCH_ERROR,
                                 cudaGetErrorString(err));
    }
    return ayaka_status_ok();
}

}  // namespace

ayaka_status_t rmsnorm_cuda(const ayaka_tensor_view_t& out,
                            const ayaka_tensor_view_t& input,
                            const ayaka_tensor_view_t& weight,
                            float eps,
                            ayaka_stream_t stream) {
    const int64_t hidden = weight.shape[0];
    int64_t rows = 1;
    for (int32_t i = 0; i + 1 < input.rank; ++i) {
        rows *= input.shape[i];
    }
    if (rows == 0 || hidden == 0) {
        return ayaka_status_ok();
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    switch (input.dtype) {
        case AYAKA_DTYPE_F32:
            return launch_rmsnorm<float>(out, input, weight, rows, hidden, eps,
                                         cuda_stream);
        case AYAKA_DTYPE_F16:
            return launch_rmsnorm<__half>(out, input, weight, rows, hidden, eps,
                                          cuda_stream);
        case AYAKA_DTYPE_BF16:
            return launch_rmsnorm<__nv_bfloat16>(out, input, weight, rows, hidden,
                                                 eps, cuda_stream);
        default:
            return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                                     "rmsnorm: dtype must be f32, f16, or bf16");
    }
}

}  // namespace ayaka
