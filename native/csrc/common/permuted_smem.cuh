#pragma once

// XOR shared-memory swizzle, matching the host index math in
// crates/ayaka-memory/src/swizzle.rs. Permuting each tile row by
// `col ^ (row & mask)` keeps a column access across a warp on 32 distinct
// banks. `cols_mask` is (cols - 1) for a power-of-two column count.

#include <stdint.h>

namespace ayaka {

__device__ __forceinline__ int xor_swizzle(int row, int col, int cols_mask) {
  return (col & cols_mask) ^ (row & cols_mask);
}

}  // namespace ayaka