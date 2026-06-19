// Naive W4A16 GEMM kernel — correctness baseline.
// Each thread computes one output element. No shared memory, no Tensor Cores.

#include "w4a16_gemm.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "ayaka/dtype.h"

namespace ayaka {

namespace {

// Naive kernel: one thread per output element.
// Grid: (ceil_div(M, 16), ceil_div(N, 16)), Block: (16, 16)
template <typename T>
__global__ void w4a16_gemm_naive_kernel(
    T* __restrict__ out,                  // [M, N]
    const T* __restrict__ a,              // [M, K]
    const uint8_t* __restrict__ b_quant,  // [N, K/2] packed 4-bit
    const T* __restrict__ b_scales,       // [N, K/group_size]
    const T* __restrict__ b_mins,         // [N, K/group_size] or nullptr
    const T* __restrict__ bias,           // [N] or nullptr
    int M, int N, int K, int group_size,
    int n_groups  // K / group_size
) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a_val = static_cast<float>(a[m * K + k]);

    int byte_idx = k >> 1;  // k / 2
    uint8_t packed = b_quant[n * (K >> 1) + byte_idx];
    int nibble = (k & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);

    int group_idx = k / group_size;
    float scale = static_cast<float>(b_scales[n * n_groups + group_idx]);

    float weight;
    if (b_mins != nullptr) {
      float min_val = static_cast<float>(b_mins[n * n_groups + group_idx]);
      weight = scale * static_cast<float>(nibble) + min_val;
    } else {
      weight = scale * static_cast<float>(nibble - 8);
    }

    acc += a_val * weight;
  }

  if (bias != nullptr) {
    acc += static_cast<float>(bias[n]);
  }

  out[m * N + n] = static_cast<T>(acc);
}

template <typename T>
ayaka_status_t launch_typed(const ayaka_tensor_view_t& out,
                            const ayaka_tensor_view_t& a,
                            const ayaka_tensor_view_t& b_quant,
                            const ayaka_tensor_view_t& b_scales,
                            const ayaka_tensor_view_t* b_mins,
                            const ayaka_tensor_view_t* bias, int32_t group_size,
                            ayaka_stream_t stream) {
  const int M = static_cast<int>(a.shape[0]);
  const int K = static_cast<int>(a.shape[1]);
  const int N = static_cast<int>(out.shape[1]);
  const int n_groups = K / group_size;

  T* out_ptr = static_cast<T*>(out.data);
  const T* a_ptr = static_cast<const T*>(a.data);
  const uint8_t* bq_ptr = static_cast<const uint8_t*>(b_quant.data);
  const T* bs_ptr = static_cast<const T*>(b_scales.data);
  const T* bm_ptr = b_mins ? static_cast<const T*>(b_mins->data) : nullptr;
  const T* bias_ptr = bias ? static_cast<const T*>(bias->data) : nullptr;

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);

  w4a16_gemm_naive_kernel<T>
      <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          out_ptr, a_ptr, bq_ptr, bs_ptr, bm_ptr, bias_ptr, M, N, K, group_size,
          n_groups);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return ayaka_status_make(AYAKA_STATUS_KERNEL_LAUNCH_ERROR,
                             cudaGetErrorString(err));
  }
  return ayaka_status_ok();
}

}  // namespace

ayaka_status_t w4a16_gemm_naive(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& a,
    const ayaka_tensor_view_t& b_quant, const ayaka_tensor_view_t& b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* /*workspace*/, size_t /*workspace_bytes*/,
    ayaka_stream_t stream) {
  if (out.dtype == AYAKA_DTYPE_F16) {
    return launch_typed<__half>(out, a, b_quant, b_scales, b_mins, bias,
                                group_size, stream);
  }
  if (out.dtype == AYAKA_DTYPE_BF16) {
    return launch_typed<__nv_bfloat16>(out, a, b_quant, b_scales, b_mins, bias,
                                       group_size, stream);
  }
  return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                           "w4a16_gemm: only F16/BF16 output supported");
}

}  // namespace ayaka