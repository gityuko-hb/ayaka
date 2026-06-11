#pragma once

// Host-side launch helpers shared by every kernel launcher.

#include <cuda_runtime.h>

#include <stdint.h>

#include "ayaka/error.h"

namespace ayaka {

// Default thread count for elementwise / row-per-block kernels.
inline constexpr int kBlockSize = 256;

// Grid cap for grid-stride kernels; keeps huge tensors from exhausting the
// launch configuration while the stride loop covers the remainder.
inline constexpr unsigned int kMaxGridBlocks = 65536;

AYAKA_NODISCARD inline constexpr int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Grid size for a grid-stride kernel over `total` elements.
AYAKA_NODISCARD inline unsigned int grid_stride_blocks(int64_t total, int block) {
    const int64_t wanted = ceil_div(total, block);
    return wanted < static_cast<int64_t>(kMaxGridBlocks)
               ? static_cast<unsigned int>(wanted)
               : kMaxGridBlocks;
}

// Status of the launch that was just enqueued. Call immediately after the
// <<<...>>> expression; only configuration/launch errors are visible here,
// asynchronous execution errors surface on a later synchronize.
AYAKA_NODISCARD inline ayaka_status_t last_launch_status() {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ayaka_status_make(AYAKA_STATUS_KERNEL_LAUNCH_ERROR,
                                 cudaGetErrorString(err));
    }
    return ayaka_status_ok();
}

}  // namespace ayaka
