#include <cuda_runtime.h>

#include <stdint.h>

#include "ayaka/ops/rope.h"
#include "common/dispatch.cuh"
#include "common/launch_utils.cuh"
#include "common/type_convert.cuh"
#include "pos/rope.h"

namespace ayaka {
namespace {

// Rotate the leading 2*rot_half elements of every head in one token row.
template <typename T, bool NEOX>
__device__ __forceinline__ void apply_rope_row(T* __restrict__ row,
                                               int num_heads,
                                               int head_dim,
                                               const float* __restrict__ cos_row,
                                               const float* __restrict__ sin_row,
                                               int rot_half) {
    const int n = num_heads * rot_half;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int head = i / rot_half;
        const int d = i % rot_half;
        T* h = row + static_cast<int64_t>(head) * head_dim;
        const float c = cos_row[d];
        const float s = sin_row[d];
        if (NEOX) {
            const float x1 = to_float(h[d]);
            const float x2 = to_float(h[d + rot_half]);
            h[d] = from_float<T>(x1 * c - x2 * s);
            h[d + rot_half] = from_float<T>(x2 * c + x1 * s);
        } else {
            const float x1 = to_float(h[2 * d]);
            const float x2 = to_float(h[2 * d + 1]);
            h[2 * d] = from_float<T>(x1 * c - x2 * s);
            h[2 * d + 1] = from_float<T>(x2 * c + x1 * s);
        }
    }
}

// One block per token; each thread owns disjoint (head, d) pairs, so the
// in-place update has no cross-thread hazards.
template <typename T, bool NEOX>
__global__ void rope_kernel(T* __restrict__ query,
                            T* __restrict__ key,
                            const int64_t* __restrict__ positions,
                            const float* __restrict__ cache,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int rot_half) {
    const int64_t token = blockIdx.x;
    const int64_t pos = positions[token];
    const float* cos_row = cache + pos * (2 * rot_half);
    const float* sin_row = cos_row + rot_half;

    apply_rope_row<T, NEOX>(query + token * num_heads * head_dim, num_heads,
                            head_dim, cos_row, sin_row, rot_half);
    apply_rope_row<T, NEOX>(key + token * num_kv_heads * head_dim, num_kv_heads,
                            head_dim, cos_row, sin_row, rot_half);
}

template <typename T>
AYAKA_NODISCARD ayaka_status_t launch_rope(const ayaka_tensor_view_t& query,
                                           const ayaka_tensor_view_t& key,
                                           const ayaka_tensor_view_t& positions,
                                           const ayaka_tensor_view_t& cache,
                                           bool neox,
                                           cudaStream_t stream) {
    const int64_t num_tokens = query.shape[0];
    const int num_heads = static_cast<int>(query.shape[1]);
    const int num_kv_heads = static_cast<int>(key.shape[1]);
    const int head_dim = static_cast<int>(query.shape[2]);
    const int rot_half = static_cast<int>(cache.shape[1] / 2);
    const dim3 grid(static_cast<unsigned int>(num_tokens));

    T* q = static_cast<T*>(query.data);
    T* k = static_cast<T*>(key.data);
    const int64_t* pos = static_cast<const int64_t*>(positions.data);
    const float* cs = static_cast<const float*>(cache.data);

    if (neox) {
        rope_kernel<T, true><<<grid, kBlockSize, 0, stream>>>(
            q, k, pos, cs, num_heads, num_kv_heads, head_dim, rot_half);
    } else {
        rope_kernel<T, false><<<grid, kBlockSize, 0, stream>>>(
            q, k, pos, cs, num_heads, num_kv_heads, head_dim, rot_half);
    }

    return last_launch_status();
}

}  // namespace

ayaka_status_t rope_cuda(const ayaka_tensor_view_t& query,
                         const ayaka_tensor_view_t& key,
                         const ayaka_tensor_view_t& positions,
                         const ayaka_tensor_view_t& cos_sin_cache,
                         int32_t layout,
                         ayaka_stream_t stream) {
    if (query.shape[0] == 0 || cos_sin_cache.shape[1] == 0) {
        return ayaka_status_ok();
    }

    const bool neox = layout == AYAKA_ROPE_LAYOUT_NEOX;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    AYAKA_DISPATCH_FLOAT_DTYPES(query.dtype, "rope", [&] {
        return launch_rope<scalar_t>(query, key, positions, cos_sin_cache, neox,
                                     cuda_stream);
    });
}

}  // namespace ayaka
