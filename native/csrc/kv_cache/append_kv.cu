#include <cuda_runtime.h>

#include <stdint.h>

#include "ayaka/ops/append_kv.h"
#include "common/dispatch.cuh"
#include "common/launch_utils.cuh"
#include "kv_cache/append_kv.h"
#include "kv_cache/kv_offset.cuh"
namespace ayaka {
namespace {
// One block per token; each thread copies disjoint (head, d) elements, so
// the scatter has no cross-thread hazards. slot < 0 marks a padded token to
// skip (e.g. positions beyond the sequence). The copy is bit-exact: K/V are
// stored in the cache verbatim with no dtype conversion.
template <typename T>
__global__ void append_kv_kernel(const T* __restrict__ key,
                                 const T* __restrict__ value,
                                 const int64_t* __restrict__ slot_mapping,
                                 T* __restrict__ kv_cache, int block_size,
                                 int H_kv, int D) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int hd = H_kv * D;
  const T* k_row = key + token * hd;
  const T* v_row = value + token * hd;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) {
    const int h = i / D;
    const int d = i % D;
    kv_cache[nhd_kv_offset(slot, 0, h, d, block_size, H_kv, D)] = k_row[i];
    kv_cache[nhd_kv_offset(slot, 1, h, d, block_size, H_kv, D)] = v_row[i];
  }
}
template <typename T>
AYAKA_NODISCARD ayaka_status_t launch_append_kv(
    const ayaka_tensor_view_t& key, const ayaka_tensor_view_t& value,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& slot_mapping, cudaStream_t stream) {
  const int64_t num_tokens = key.shape[0];
  const int block_size = static_cast<int>(kv_cache.shape[2]);
  const int H_kv = static_cast<int>(kv_cache.shape[3]);
  const int D = static_cast<int>(kv_cache.shape[4]);
  const dim3 grid(static_cast<unsigned int>(num_tokens));
  const T* k = static_cast<const T*>(key.data);
  const T* v = static_cast<const T*>(value.data);
  const int64_t* slots = static_cast<const int64_t*>(slot_mapping.data);
  T* cache = static_cast<T*>(kv_cache.data);
  append_kv_kernel<T><<<grid, kBlockSize, 0, stream>>>(k, v, slots, cache,
                                                       block_size, H_kv, D);
  return last_launch_status();
}
}  // namespace
ayaka_status_t append_kv_cuda(const ayaka_tensor_view_t& key,
                              const ayaka_tensor_view_t& value,
                              const ayaka_tensor_view_t& kv_cache,
                              const ayaka_tensor_view_t& slot_mapping,
                              int32_t layout, ayaka_stream_t stream) {
  if (key.shape[0] == 0) {
    return ayaka_status_ok();
  }
  if (layout != AYAKA_KV_LAYOUT_NHD) {
    return ayaka_status_make(AYAKA_STATUS_UNSUPPORTED,
                             "append_kv: only NHD layout is implemented");
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  AYAKA_DISPATCH_FLOAT_DTYPES(key.dtype, "append_kv", [&] {
    return launch_append_kv<scalar_t>(key, value, kv_cache, slot_mapping,
                                      cuda_stream);
  });
}
}  // namespace ayaka