#pragma once

// Element-offset arithmetic for the paged KV cache. Header-only and shared
// between the write path (append_kv) and the future attention read path.

#include <stdint.h>

namespace ayaka {

// Flat element offset of (slot, kv, h, d) in an NHD paged KV cache shaped
// [num_blocks, 2, block_size, H_kv, D], where dim 1 selects K (kv=0) or
// V (kv=1). `slot` indexes a (block, in-block position) pair:
//   block = slot / block_size, s = slot % block_size.
__device__ __forceinline__ int64_t nhd_kv_offset(int64_t slot, int kv, int h,
                                                 int d, int block_size,
                                                 int H_kv, int D) {
  const int64_t block = slot / block_size;
  const int64_t s = slot % block_size;
  return (((block * 2 + kv) * block_size + s) * H_kv + h) *
             static_cast<int64_t>(D) +
         d;
}
}  // namespace ayaka