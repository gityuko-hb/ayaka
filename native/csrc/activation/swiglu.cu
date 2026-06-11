#include <cuda_runtime.h>

#include <stdint.h>

#include "activation/swiglu.h"
#include "common/dispatch.cuh"
#include "common/launch_utils.cuh"
#include "common/type_convert.cuh"
#include "ops/common.h"

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
AYAKA_NODISCARD ayaka_status_t launch_silu_and_mul(const ayaka_tensor_view_t& out,
                                                   const ayaka_tensor_view_t& input,
                                                   int64_t hidden,
                                                   int64_t total,
                                                   cudaStream_t stream) {
    const unsigned int blocks = grid_stride_blocks(total, kBlockSize);

    silu_and_mul_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
        static_cast<T*>(out.data),
        static_cast<const T*>(input.data),
        hidden,
        total);

    return last_launch_status();
}

}  // namespace

ayaka_status_t silu_and_mul_cuda(const ayaka_tensor_view_t& out,
                                 const ayaka_tensor_view_t& input,
                                 ayaka_stream_t stream) {
    const int64_t hidden = out.shape[out.rank - 1];
    const int64_t total = view_numel(&out);
    if (total == 0) {
        return ayaka_status_ok();
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    AYAKA_DISPATCH_FLOAT_DTYPES(input.dtype, "silu_and_mul", [&] {
        return launch_silu_and_mul<scalar_t>(out, input, hidden, total,
                                             cuda_stream);
    });
}

}  // namespace ayaka
