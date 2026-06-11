#pragma once

// Block-wide reductions shared by normalization (and later online-softmax)
// kernels.

namespace ayaka {

// Sum `v` across the block; every thread returns the total.
// Uses two static __shared__ buffers; callers must not assume any other
// shared memory survives across this call.
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

}  // namespace ayaka
