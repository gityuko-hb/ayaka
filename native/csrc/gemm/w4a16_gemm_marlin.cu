// Marlin-style W4A16 GEMM kernel
//
// Simplified Marlin path for SM80+ that uses three core Marlin techniques:
//   1. Offline weight shuffle (Rust `repack_to_marlin`) -> `[N/16, K/16, 16,
//   8]`
//      so a warp can stream contiguous bytes for one tile.
//   2. Register-level 4-bit unpacking via PTX `prmt.b32` to extract nibbles
//      straight into registers (no shared-mem detour for dequantized weights).
//   3. `cp.async` double buffering of weight tiles: prefetch tile k+1 into
//      shared memory while computing tile k.
//
// Accumulation is f32 FMA for now; Tensor Core `mma.sync.m16n8k16` is a
// follow-up optimization. This still cuts global-memory traffic versus the
// naive baseline because each 16x16 weight tile is loaded once per block and
// reused across all M rows handled by the block.

#include "w4a16_gemm_marlin.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "ayaka/dtype.h"
#include "common/launch_utils.cuh"
#include "common/type_convert.cuh"
#include "gemm/marlin_shuffle.cuh"

namespace ayaka {
namespace {

// Marlin thread block: 128 threads, each owns one M row and accumulates 16
// output columns (one N-tile). Grid: (ceil_div(M, 128), N/16).
constexpr int kMarlinThreads = 128;
constexpr int kMarlinSmemBytes = 2 * MARLIN_TILE_BYTES;  // double buffer

// Unpack 8 nibbles (4 bytes) from a u32 into 8 f32 dequantized weights using
// PTX `prmt.b32` to broadcast each byte into a register, then mask off the
// low/high nibble. `prmt.b32 d, a, 0, ctrl` replicates byte (ctrl>>4 lane) of
// `a` across `d` when ctrl is 0xSSSS with S in 0..3 selecting bytes 0..3.
//
// `scale`/`min_val` are f32 (converted from the stored f16 on the caller
// side) so dequant matches the naive kernel bit-for-bit; the f16-in-registers
// form is reserved for the future `mma.sync` path.
template <bool kHasMin>
__device__ __forceinline__ void unpack_q4_to_f32(uint32_t packed, float scale,
                                                 float min_val, float* out) {
  uint32_t b0, b1, b2, b3;
  asm("prmt.b32 %0, %1, 0, 0x0000;" : "=r"(b0) : "r"(packed));  // byte 0
  asm("prmt.b32 %0, %1, 0, 0x1111;" : "=r"(b1) : "r"(packed));  // byte 1
  asm("prmt.b32 %0, %1, 0, 0x2222;" : "=r"(b2) : "r"(packed));  // byte 2
  asm("prmt.b32 %0, %1, 0, 0x3333;" : "=r"(b3) : "r"(packed));  // byte 3

  uint32_t bytes[4] = {b0, b1, b2, b3};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Nibble i lives in byte i/2: low nibble if i even, high nibble if odd.
    uint32_t by = bytes[i >> 1];
    int nibble = (i & 1) ? int((by >> 4) & 0x0Fu) : int(by & 0x0Fu);
    out[i] = kHasMin ? (scale * static_cast<float>(nibble) + min_val)
                     : (scale * static_cast<float>(nibble - 8));
  }
}

// Copy one 128-byte Marlin weight tile from global to shared memory using
// 32 x 4-byte `cp.async.ca` copies. `commit_group` is issued by the caller so
// the prefetch can be grouped with other work.
__device__ __forceinline__ void cp_async_load_tile(
    uint8_t* __restrict__ smem_dst, const uint8_t* __restrict__ gmem_src) {
  const int tid = threadIdx.x;
  for (int i = tid; i < MARLIN_TILE_BYTES / 4; i += kMarlinThreads) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst + i * 4);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr),
                 "l"(gmem_src + i * 4));
  }
}

template <typename T, bool kHasMin>
__global__ void w4a16_gemm_marlin_kernel(
    T* __restrict__ out,                  // [M, N]
    const T* __restrict__ a,              // [M, K]
    const uint8_t* __restrict__ b_quant,  // Marlin [N/16, K/16, 16, 8]
    const T* __restrict__ b_scales,       // [N, K/group_size]
    const T* __restrict__ b_mins,         // [N, K/group_size] or nullptr
    const T* __restrict__ bias,           // [N] or nullptr
    int M, int N, int K, int group_size, int n_groups, int n_tiles_k) {
  extern __shared__ uint8_t smem[];
  uint8_t* smem_b[2];
  smem_b[0] = smem;
  smem_b[1] = smem + MARLIN_TILE_BYTES;

  const int m = blockIdx.x * kMarlinThreads + threadIdx.x;
  const int tile_n = blockIdx.y;  // each block owns one 16-wide N-tile
  const int n_base = tile_n * 16;
  const bool active = (m < M);

  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

  // Prefetch tile 0 into buffer 0 and wait so the first iteration can read.
  cp_async_load_tile(smem_b[0],
                     b_quant + (tile_n * n_tiles_k) * MARLIN_TILE_BYTES);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();

  for (int tk = 0; tk < n_tiles_k; ++tk) {
    const int buf = tk & 1;

    // Prefetch the next tile into the opposite buffer.
    if (tk + 1 < n_tiles_k) {
      const int next_tile = tile_n * n_tiles_k + (tk + 1);
      cp_async_load_tile(smem_b[buf ^ 1],
                         b_quant + next_tile * MARLIN_TILE_BYTES);
      asm volatile("cp.async.commit_group;");
    }

    // Precompute the 16 activations for this K-tile (shared across N rows).
    float a_vals[16];
    if (active) {
#pragma unroll
      for (int k_in = 0; k_in < 16; ++k_in) {
        a_vals[k_in] = to_float(a[m * K + tk * 16 + k_in]);
      }
    }

    // group_size is a multiple of 16, so the whole 16-wide K-tile shares
    // one group index.
    const int group_idx = (tk * 16) / group_size;

// Walk the 16 N rows of the tile; unpack each row's 16 nibbles in
// registers and FMA into the per-column accumulators.
#pragma unroll
    for (int n_in = 0; n_in < 16; ++n_in) {
      const int n = n_base + n_in;
      const float scale = to_float(b_scales[n * n_groups + group_idx]);
      const float min_val =
          kHasMin ? to_float(b_mins[n * n_groups + group_idx]) : 0.0f;

      const uint32_t lo = *reinterpret_cast<const uint32_t*>(
          &smem_b[buf][n_in * MARLIN_BYTES_PER_TILE_ROW]);
      const uint32_t hi = *reinterpret_cast<const uint32_t*>(
          &smem_b[buf][n_in * MARLIN_BYTES_PER_TILE_ROW + 4]);

      float w[16];
      unpack_q4_to_f32<kHasMin>(lo, scale, min_val, w);
      unpack_q4_to_f32<kHasMin>(hi, scale, min_val, w + 8);

      if (active) {
#pragma unroll
        for (int k_in = 0; k_in < 16; ++k_in) {
          acc[n_in] = fmaf(a_vals[k_in], w[k_in], acc[n_in]);
        }
      }
    }

    // Wait for the next-tile prefetch before the next iteration reads it.
    if (tk + 1 < n_tiles_k) {
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
    }
  }

  if (active) {
#pragma unroll
    for (int n_in = 0; n_in < 16; ++n_in) {
      const int n = n_base + n_in;
      float v = acc[n_in];
      if (bias != nullptr) v += to_float(bias[n]);
      out[m * N + n] = from_float<T>(v);
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

  dim3 block(kMarlinThreads);
  dim3 grid(
      static_cast<unsigned int>((M + kMarlinThreads - 1) / kMarlinThreads),
      static_cast<unsigned int>(N / MARLIN_TILE_N));

  w4a16_gemm_marlin_kernel<T, kHasMin>
      <<<grid, block, kMarlinSmemBytes, static_cast<cudaStream_t>(stream)>>>(
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
