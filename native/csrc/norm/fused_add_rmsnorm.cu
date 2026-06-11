#include <cuda_runtime.h>

#include <stdint.h>

#include "common/reduction.cuh"
#include "common/type_convert.cuh"
#include "norm/fused_add_rmsnorm.h"

namespace ayaka {
namespace {

// One block per row. Pass 1 writes input+residual into residual_out while
// accumulating the f32 sum of squares; pass 2 re-reads each thread's own
// residual_out elements (no cross-thread reads, so no extra barrier beyond
// the one inside block_reduce_sum) and writes the normalized result.
// Safe when out aliases input and/or residual_out aliases residual.
template <typename T, int BLOCK>
__global__ void fused_add_rmsnorm_kernel(T* __restrict__ out,
                                         T* __restrict__ residual_out,
                                         const T* __restrict__ input,
                                         const T* __restrict__ residual,
                                         const T* __restrict__ weight,
                                         int64_t hidden,
                                         float eps) {
    const int64_t row = blockIdx.x;
    const T* row_in = input + row * hidden;
    const T* row_res = residual + row * hidden;
    T* row_res_out = residual_out + row * hidden;
    T* row_out = out + row * hidden;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
        const float v = to_float(row_in[i]) + to_float(row_res[i]);
        row_res_out[i] = from_float<T>(v);
        sum_sq = fmaf(v, v, sum_sq);
    }

    const float total = block_reduce_sum<BLOCK>(sum_sq);
    const float inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);

    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
        const float v = to_float(row_res_out[i]) * inv_rms * to_float(weight[i]);
        row_out[i] = from_float<T>(v);
    }
}

template <typename T>
ayaka_status_t launch_fused_add_rmsnorm(const ayaka_tensor_view_t& out,
                                        const ayaka_tensor_view_t& residual_out,
                                        const ayaka_tensor_view_t& input,
                                        const ayaka_tensor_view_t& residual,
                                        const ayaka_tensor_view_t& weight,
                                        int64_t rows,
                                        int64_t hidden,
                                        float eps,
                                        cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const dim3 grid(static_cast<unsigned int>(rows));

    fused_add_rmsnorm_kernel<T, BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<T*>(out.data),
        static_cast<T*>(residual_out.data),
        static_cast<const T*>(input.data),
        static_cast<const T*>(residual.data),
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

ayaka_status_t fused_add_rmsnorm_cuda(const ayaka_tensor_view_t& out,
                                      const ayaka_tensor_view_t& residual_out,
                                      const ayaka_tensor_view_t& input,
                                      const ayaka_tensor_view_t& residual,
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
            return launch_fused_add_rmsnorm<float>(out, residual_out, input,
                                                   residual, weight, rows, hidden,
                                                   eps, cuda_stream);
        case AYAKA_DTYPE_F16:
            return launch_fused_add_rmsnorm<__half>(out, residual_out, input,
                                                    residual, weight, rows, hidden,
                                                    eps, cuda_stream);
        case AYAKA_DTYPE_BF16:
            return launch_fused_add_rmsnorm<__nv_bfloat16>(
                out, residual_out, input, residual, weight, rows, hidden, eps,
                cuda_stream);
        default:
            return ayaka_status_make(
                AYAKA_STATUS_UNSUPPORTED,
                "fused_add_rmsnorm: dtype must be f32, f16, or bf16");
    }
}

}  // namespace ayaka
