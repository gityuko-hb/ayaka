// Marlin-style W4A16 GEMM kernel
//
// Simplified Marlin path for SM80+ that uses the Marlin weight shuffle layout
// `[N/16, K/16, 16, 8]` (offline `repack_to_marlin`) so a warp can stream
// contiguous bytes for one 16x16 weight tile.
//
// The compute core is `nvcuda::wmma` Tensor Cores (16x16x16, f32 accumulate).
// Each warp owns one 16x16 output tile: per K-step it dequantizes one 16x16
// weight tile into shared memory, loads A/B fragments, and calls `mma_sync`.
// There is no double buffering; the single-warp tile loop keeps the path
// simple while still loading each weight tile once per output tile.

#include "w4a16_gemm_marlin.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <mma.h>

#include "ayaka/dtype.h"
#include "common/launch_utils.cuh"
#include "common/type_convert.cuh"
#include "gemm/marlin_shuffle.cuh"

namespace ayaka {
namespace {

using namespace nvcuda;

// One warp computes one 16x16 output tile; block = 1 warp.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;                   // == MARLIN_TILE_K
constexpr int kTileElems = WMMA_M * WMMA_K;  // 256
constexpr int kWarpThreads = 32;

template <typename T, bool kHasMin>
__global__ void w4a16_gemm_marlin_kernel(
    T* __restrict__ out,                  // [M, N]
    const T* __restrict__ a,              // [M, K]
    const uint8_t* __restrict__ b_quant,  // Marlin [N/16, K/16, 16, 8]
    const T* __restrict__ b_scales,       // [N, K/group_size]
    const T* __restrict__ b_mins,         // [N, K/group_size] or nullptr
    const T* __restrict__ bias,           // [N] or nullptr
    int M, int N, int K, int group_size, int n_groups, int n_tiles_k) {
  const int m0 = blockIdx.x * WMMA_M;
  const int n0 = blockIdx.y * WMMA_N;
  const int tile_n = blockIdx.y;
  const int lane = threadIdx.x;  // 0..31

  __shared__ T a_smem[kTileElems];      // [16(M)][16(K)] row-major
  __shared__ T w_smem[kTileElems];      // [16(N)][16(K)] row-major
  __shared__ float c_smem[kTileElems];  // [16(M)][16(N)] row-major

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int tk = 0; tk < n_tiles_k; ++tk) {
    const int k0 = tk * WMMA_K;
    const int group_idx = k0 / group_size;  // group_size % 16 == 0

    // Stage A tile [16(M)][16(K)] into smem, zero-padding rows >= M.
    for (int i = lane; i < kTileElems; i += kWarpThreads) {
      const int mr = i / WMMA_K;
      const int kc = i % WMMA_K;
      const int gm = m0 + mr;
      a_smem[i] = (gm < M) ? a[gm * K + k0 + kc] : from_float<T>(0.0f);
    }

    // Dequant the Marlin weight tile (tile_n, tk) into w_smem[nr*16 + kc] =
    // W[n, k].
    const uint8_t* tile =
        b_quant +
        static_cast<size_t>(tile_n * n_tiles_k + tk) * MARLIN_TILE_BYTES;
    for (int i = lane; i < kTileElems; i += kWarpThreads) {
      const int nr = i / WMMA_K;
      const int kc = i % WMMA_K;
      const uint8_t byte = tile[nr * MARLIN_BYTES_PER_TILE_ROW + (kc >> 1)];
      const int nibble = (kc & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
      const int n = n0 + nr;
      const float scale = to_float(b_scales[n * n_groups + group_idx]);
      float w;
      if (kHasMin) {
        const float mn = to_float(b_mins[n * n_groups + group_idx]);
        w = scale * static_cast<float>(nibble) + mn;
      } else {
        w = scale * static_cast<float>(nibble - 8);
      }
      w_smem[i] = from_float<T>(w);
    }
    __syncwarp();

    // matrix_b col_major, ldm=16 reads ptr[k + n*16] = w_smem[n*16 + k] =
    // W[n,k], i.e. B[k,n] = W[n,k], so C = A @ W^T (matches quant_gemm_ref).
    wmma::load_matrix_sync(a_frag, a_smem, WMMA_K);
    wmma::load_matrix_sync(b_frag, w_smem, WMMA_K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(c_smem, c_frag, WMMA_N, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < kTileElems; i += kWarpThreads) {
    const int mr = i / WMMA_N;
    const int nc = i % WMMA_N;
    const int gm = m0 + mr;
    if (gm < M) {
      float v = c_smem[mr * WMMA_N + nc];
      const int n = n0 + nc;
      if (bias != nullptr) v += to_float(bias[n]);
      out[gm * N + n] = from_float<T>(v);
    }
  }
}

template <typename T, bool kHasMin>
AYAKA_NODISCARD ayaka_status_t launch_marlin(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& a,
    const ayaka_tensor_view_t& b_quant, const ayaka_tensor_view_t& b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, ayaka_stream_t stream) {
  const int M = static_cast<int>(a.shape[0]);
  const int K = static_cast<int>(a.shape[1]);
  const int N = static_cast<int>(out.shape[1]);
  const int n_groups = K / group_size;
  const int n_tiles_k = K / MARLIN_TILE_K;

  T* out_ptr = static_cast<T*>(out.data);
  const T* a_ptr = static_cast<const T*>(a.data);
  const uint8_t* bq_ptr = static_cast<const uint8_t*>(b_quant.data);
  const T* bs_ptr = static_cast<const T*>(b_scales.data);
  const T* bm_ptr = b_mins ? static_cast<const T*>(b_mins->data) : nullptr;
  const T* bias_ptr = bias ? static_cast<const T*>(bias->data) : nullptr;

  dim3 block(kWarpThreads);
  dim3 grid(static_cast<unsigned int>((M + WMMA_M - 1) / WMMA_M),
            static_cast<unsigned int>(N / WMMA_N));

  w4a16_gemm_marlin_kernel<T, kHasMin>
      <<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          out_ptr, a_ptr, bq_ptr, bs_ptr, bm_ptr, bias_ptr, M, N, K, group_size,
          n_groups, n_tiles_k);

  return last_launch_status();
}

}  // namespace

ayaka_status_t w4a16_gemm_marlin(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& a,
    const ayaka_tensor_view_t& b_quant, const ayaka_tensor_view_t& b_scales,
    const ayaka_tensor_view_t* b_mins, const ayaka_tensor_view_t* bias,
    int32_t group_size, void* /*workspace*/, size_t /*workspace_bytes*/,
    ayaka_stream_t stream) {
  // SM80+ gate: `cp.async` double buffering targets Ampere Tensor Core era.
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             out.device_ordinal) != cudaSuccess) {
    return ayaka_status_make(AYAKA_STATUS_DEVICE_ERROR,
                             "w4a16_gemm_marlin: could not query device SM");
  }
  if (major < 8) {
    return ayaka_status_make(
        AYAKA_STATUS_UNSUPPORTED,
        "w4a16_gemm_marlin: requires SM80+ (Ampere or newer)");
  }

  const int64_t N = out.shape[1];
  const int64_t K = a.shape[1];
  if (N % MARLIN_TILE_N != 0 || K % MARLIN_TILE_K != 0) {
    return ayaka_status_make(
        AYAKA_STATUS_UNSUPPORTED,
        "w4a16_gemm_marlin: N and K must be multiples of 16");
  }
  if (group_size % MARLIN_TILE_K != 0) {
    return ayaka_status_make(
        AYAKA_STATUS_UNSUPPORTED,
        "w4a16_gemm_marlin: group_size must be a multiple of 16");
  }

  const bool has_min = (b_mins != nullptr);

  if (out.dtype == AYAKA_DTYPE_F16) {
    if (has_min) {
      return launch_marlin<__half, true>(out, a, b_quant, b_scales, b_mins,
                                         bias, group_size, stream);
    }
    return launch_marlin<__half, false>(out, a, b_quant, b_scales, b_mins, bias,
                                        group_size, stream);
  }
  if (out.dtype == AYAKA_DTYPE_BF16) {
    if (has_min) {
      return launch_marlin<__nv_bfloat16, true>(
          out, a, b_quant, b_scales, b_mins, bias, group_size, stream);
    }
    return launch_marlin<__nv_bfloat16, false>(
        out, a, b_quant, b_scales, b_mins, bias, group_size, stream);
  }
  return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                           "w4a16_gemm_marlin: only F16/BF16 output supported");
}

}  // namespace ayaka
