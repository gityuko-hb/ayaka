#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

#include "attention/paged_attention.h"
#include "ayaka/ops/paged_attention.h"
#include "common/dispatch.cuh"
#include "common/launch_utils.cuh"
#include "common/reduction.cuh"
#include "common/type_convert.cuh"
#include "kv_cache/kv_offset.cuh"

namespace ayaka {
namespace {

// Decode paged attention (one query token per sequence). One block per
// (sequence, query head); BLOCK threads stride over head_dim. KV is read from
// the NHD paged cache via nhd_kv_offset, gathered through block_tables. Scores
// and the value accumulation use a numerically-stable online softmax in f32,
// so storage dtype never limits accuracy. Dynamic shared memory holds the
// query row and the running output accumulator (2 * head_dim floats).
template <typename T, int BLOCK>
__global__ void paged_attention_decode_kernel(
    T* __restrict__ out, const T* __restrict__ query,
    const T* __restrict__ kv_cache, const int64_t* __restrict__ block_tables,
    const int64_t* __restrict__ seq_lens, float scale, int num_heads,
    int num_kv_heads, int head_dim, int block_size, int max_num_blocks) {
  extern __shared__ float smem[];
  float* q_sh = smem;            // [head_dim]
  float* acc = smem + head_dim;  // [head_dim]
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int group = num_heads / num_kv_heads;  // GQA fan-out
  const int kv_head = head / group;
  const int64_t seq_len = seq_lens[seq];
  const T* q_row =
      query + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
    q_sh[d] = to_float(q_row[d]);
    acc[d] = 0.0f;
  }
  __syncthreads();
  float m = -INFINITY;  // running max logit
  float l = 0.0f;       // running denominator
  const int64_t* bt = block_tables + static_cast<int64_t>(seq) * max_num_blocks;
  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t block = bt[t / block_size];
    const int64_t slot = block * block_size + (t % block_size);
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
      const int64_t koff = nhd_kv_offset(slot, 0, kv_head, d, block_size,
                                         num_kv_heads, head_dim);
      partial += q_sh[d] * to_float(kv_cache[koff]);
    }
    const float score = scale * block_reduce_sum<BLOCK>(partial);
    const float m_new = fmaxf(m, score);
    const float corr = __expf(m - m_new);  // exp(-inf)=0 on the first step
    const float p = __expf(score - m_new);
    l = l * corr + p;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
      const int64_t voff = nhd_kv_offset(slot, 1, kv_head, d, block_size,
                                         num_kv_heads, head_dim);
      acc[d] = acc[d] * corr + p * to_float(kv_cache[voff]);
    }
    m = m_new;
  }
  T* o_row = out + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
    o_row[d] = from_float<T>(acc[d] * inv_l);
  }
}

template <typename T>
AYAKA_NODISCARD ayaka_status_t launch_paged_attention_decode(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, float scale, cudaStream_t stream) {
  const int num_seqs = static_cast<int>(query.shape[0]);
  const int num_heads = static_cast<int>(query.shape[1]);
  const int head_dim = static_cast<int>(query.shape[2]);
  const int block_size = static_cast<int>(kv_cache.shape[2]);
  const int num_kv_heads = static_cast<int>(kv_cache.shape[3]);
  const int max_num_blocks = static_cast<int>(block_tables.shape[1]);
  const dim3 grid(static_cast<unsigned int>(num_seqs),
                  static_cast<unsigned int>(num_heads));
  const size_t smem_bytes = static_cast<size_t>(2 * head_dim) * sizeof(float);
  paged_attention_decode_kernel<T, kBlockSize>
      <<<grid, kBlockSize, smem_bytes, stream>>>(
          static_cast<T*>(out.data), static_cast<const T*>(query.data),
          static_cast<const T*>(kv_cache.data),
          static_cast<const int64_t*>(block_tables.data),
          static_cast<const int64_t*>(seq_lens.data), scale, num_heads,
          num_kv_heads, head_dim, block_size, max_num_blocks);
  return last_launch_status();
}

}  // namespace

ayaka_status_t paged_attention_decode_cuda(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, float scale, int32_t layout,
    ayaka_stream_t stream) {
  if (query.shape[0] == 0) {
    return ayaka_status_ok();
  }
  if (layout != AYAKA_KV_LAYOUT_NHD) {
    return ayaka_status_make(
        AYAKA_STATUS_UNSUPPORTED,
        "paged_attention_decode: only NHD layout is implemented");
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  AYAKA_DISPATCH_FLOAT_DTYPES(query.dtype, "paged_attention_decode", [&] {
    return launch_paged_attention_decode<scalar_t>(
        out, query, kv_cache, block_tables, seq_lens, scale, cuda_stream);
  });
}

}  // namespace ayaka