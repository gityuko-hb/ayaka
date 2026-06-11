#include <cuda_runtime.h>

#include <stdint.h>

#include "activation/swiglu.h"
#include "common/type_convert.cuh"

namespace ayaka {
namespace {

// Grid-stride over rows * hidden output elements; silu computed in f32.
template <typename T>
__global__ void silu_and_mul_kernel(T* __restrict__ out,
                                    const T* __restrict__ input,
                                    int64_t hidden,
                                    int64_t total) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int64_t row = idx / hidden;
        const int64_t d = idx % hidden;
        const T* row_in = input + row * 2 * hidden;
        const float gate = to_float(row_in[d]);
        const float up = to_float(row_in[hidden + d]);
        const float silu = gate / (1.0f + expf(-gate));
        out[idx] = from_float<T>(silu * up);
    }
}

template <typename T>
ayaka_status_t launch_silu_and_mul(const ayaka_tensor_view_t& out,
                                   const ayaka_tensor_view_t& input,
                                   int64_t hidden,
                                   int64_t total,
                                   cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const int64_t wanted = (total + BLOCK - 1) / BLOCK;
    const unsigned int blocks =
        static_cast<unsigned int>(wanted < 65536 ? wanted : 65536);

    silu_and_mul_kernel<T><<<blocks, BLOCK, 0, stream>>>(
        static_cast<T*>(out.data),
        static_cast<const T*>(input.data),
        hidden,
        total);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ayaka_status_make(AYAKA_STATUS_KERNEL_LAUNCH_ERROR,
                                 cudaGetErrorString(err));
    }
    return ayaka_status_ok();
}

}  // namespace

ayaka_status_t silu_and_mul_cuda(const ayaka_tensor_view_t& out,
                                 const ayaka_tensor_view_t& input,
                                 ayaka_stream_t stream) {
    const int64_t hidden = out.shape[out.rank - 1];
    int64_t total = 1;
    for (int32_t i = 0; i < out.rank; ++i) {
        total *= out.shape[i];
    }
    if (total == 0) {
        return ayaka_status_ok();
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    switch (input.dtype) {
        case AYAKA_DTYPE_F32:
            return launch_silu_and_mul<float>(out, input, hidden, total,
                                              cuda_stream);
        case AYAKA_DTYPE_F16:
            return launch_silu_and_mul<__half>(out, input, hidden, total,
                                               cuda_stream);
        case AYAKA_DTYPE_BF16:
            return launch_silu_and_mul<__nv_bfloat16>(out, input, hidden, total,
                                                      cuda_stream);
        default:
            return ayaka_status_make(
                AYAKA_STATUS_UNSUPPORTED,
                "silu_and_mul: dtype must be f32, f16, or bf16");
    }
}

}  // namespace ayaka
