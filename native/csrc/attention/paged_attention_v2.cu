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
// Split-KV decode (vLLM v2 style): each sequence's context is partitioned into
// chunks of PARTITION_SIZE tokens; one block computes a partial softmax over
// its chunk, then a second kernel recombines the partials with a log-sum-exp.
// This keeps enough blocks in flight for long contexts / few sequences, where
// v1 (one block per (seq,head)) under-occupies the GPU.
constexpr int kPartitionSize = 512;
// Phase 1: per-(seq, head, partition) partial attention.
//   max_logits[s,h,p] = max score in the partition
//   exp_sums[s,h,p]   = sum exp(score - max) over the partition
//   tmp_out[s,h,p,:]  = sum exp(score - max) * v   (un-normalized)
template <typename T, int BLOCK>
__global__ void paged_attention_v2_partition_kernel(
    float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads, max_partitions]
    T* __restrict__ tmp_out,  // [num_seqs, num_heads, max_partitions, head_dim]
    const T* __restrict__ query, const T* __restrict__ kv_cache,
    const int64_t* __restrict__ block_tables,
    const int64_t* __restrict__ seq_lens, float scale, int num_heads,
    int num_kv_heads, int head_dim, int block_size, int max_num_blocks,
    int max_partitions) {
  extern __shared__ float smem[];
  float* q_sh = smem;            // [head_dim]
  float* acc = smem + head_dim;  // [head_dim]
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int part = blockIdx.z;
  const int64_t seq_len = seq_lens[seq];
  const int64_t start = static_cast<int64_t>(part) * kPartitionSize;
  const int64_t base_part =
      (static_cast<int64_t>(seq) * num_heads + head) * max_partitions + part;
  if (start >= seq_len) {
    if (threadIdx.x == 0) {
      max_logits[base_part] = -INFINITY;
      exp_sums[base_part] = 0.0f;
    }
    return;
  }
  const int64_t end = min(start + kPartitionSize, seq_len);
  const int group = num_heads / num_kv_heads;
  const int kv_head = head / group;
  const T* q_row =
      query + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
    q_sh[d] = to_float(q_row[d]);
    acc[d] = 0.0f;
  }
  __syncthreads();
  float m = -INFINITY;
  float l = 0.0f;
  const int64_t* bt = block_tables + static_cast<int64_t>(seq) * max_num_blocks;
  for (int64_t t = start; t < end; ++t) {
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
    const float corr = __expf(m - m_new);
    const float p = __expf(score - m_new);
    l = l * corr + p;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
      const int64_t voff = nhd_kv_offset(slot, 1, kv_head, d, block_size,
                                         num_kv_heads, head_dim);
      acc[d] = acc[d] * corr + p * to_float(kv_cache[voff]);
    }
    m = m_new;
  }
  if (threadIdx.x == 0) {
    max_logits[base_part] = m;
    exp_sums[base_part] = l;
  }
  T* out_row = tmp_out + base_part * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
    out_row[d] = from_float<T>(acc[d]);  // un-normalized; reduce divides by l
  }
}

// Phase 2: recombine the partitions for each (seq, head).
template <typename T, int BLOCK>
__global__ void paged_attention_v2_reduce_kernel(
    T* __restrict__ out, const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits, const T* __restrict__ tmp_out,
    const int64_t* __restrict__ seq_lens, int num_heads, int head_dim,
    int max_partitions) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t seq_len = seq_lens[seq];
  const int num_parts =
      static_cast<int>((seq_len + kPartitionSize - 1) / kPartitionSize);
  const int64_t base =
      (static_cast<int64_t>(seq) * num_heads + head) * max_partitions;
  T* o_row = out + (static_cast<int64_t>(seq) * num_heads + head) * head_dim;
  if (num_parts <= 0) {
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
      o_row[d] = from_float<T>(0.0f);
    }
    return;
  }
  // Global max and combined denominator across partitions.
  float global_max = -INFINITY;
  for (int p = 0; p < num_parts; ++p) {
    global_max = fmaxf(global_max, max_logits[base + p]);
  }
  float denom = 0.0f;
  for (int p = 0; p < num_parts; ++p) {
    denom += exp_sums[base + p] * __expf(max_logits[base + p] - global_max);
  }
  const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
    float v = 0.0f;
    for (int p = 0; p < num_parts; ++p) {
      const float w = __expf(max_logits[base + p] - global_max);
      v += w * to_float(tmp_out[(base + p) * head_dim + d]);
    }
    o_row[d] = from_float<T>(v * inv_denom);
  }
}
template <typename T>
AYAKA_NODISCARD ayaka_status_t launch_paged_attention_v2(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, const ayaka_tensor_view_t& exp_sums,
    const ayaka_tensor_view_t& max_logits, const ayaka_tensor_view_t& tmp_out,
    float scale, cudaStream_t stream) {
  const int num_seqs = static_cast<int>(query.shape[0]);
  const int num_heads = static_cast<int>(query.shape[1]);
  const int head_dim = static_cast<int>(query.shape[2]);
  const int block_size = static_cast<int>(kv_cache.shape[2]);
  const int num_kv_heads = static_cast<int>(kv_cache.shape[3]);
  const int max_num_blocks = static_cast<int>(block_tables.shape[1]);
  const int max_partitions = static_cast<int>(exp_sums.shape[2]);
  const dim3 part_grid(static_cast<unsigned int>(num_seqs),
                       static_cast<unsigned int>(num_heads),
                       static_cast<unsigned int>(max_partitions));
  const size_t smem_bytes = static_cast<size_t>(2 * head_dim) * sizeof(float);
  paged_attention_v2_partition_kernel<T, kBlockSize>
      <<<part_grid, kBlockSize, smem_bytes, stream>>>(
          static_cast<float*>(exp_sums.data),
          static_cast<float*>(max_logits.data), static_cast<T*>(tmp_out.data),
          static_cast<const T*>(query.data),
          static_cast<const T*>(kv_cache.data),
          static_cast<const int64_t*>(block_tables.data),
          static_cast<const int64_t*>(seq_lens.data), scale, num_heads,
          num_kv_heads, head_dim, block_size, max_num_blocks, max_partitions);
  const dim3 reduce_grid(static_cast<unsigned int>(num_seqs),
                         static_cast<unsigned int>(num_heads));
  paged_attention_v2_reduce_kernel<T, kBlockSize>
      <<<reduce_grid, kBlockSize, 0, stream>>>(
          static_cast<T*>(out.data), static_cast<const float*>(exp_sums.data),
          static_cast<const float*>(max_logits.data),
          static_cast<const T*>(tmp_out.data),
          static_cast<const int64_t*>(seq_lens.data), num_heads, head_dim,
          max_partitions);
  return last_launch_status();
}
}  // namespace

int paged_attention_partition_size() { return kPartitionSize; }
ayaka_status_t paged_attention_v2_cuda(
    const ayaka_tensor_view_t& out, const ayaka_tensor_view_t& query,
    const ayaka_tensor_view_t& kv_cache,
    const ayaka_tensor_view_t& block_tables,
    const ayaka_tensor_view_t& seq_lens, const ayaka_tensor_view_t& exp_sums,
    const ayaka_tensor_view_t& max_logits, const ayaka_tensor_view_t& tmp_out,
    float scale, int32_t layout, ayaka_stream_t stream) {
  if (query.shape[0] == 0) {
    return ayaka_status_ok();
  }
  if (layout != AYAKA_KV_LAYOUT_NHD) {
    return ayaka_status_make(
        AYAKA_STATUS_UNSUPPORTED,
        "paged_attention_v2: only NHD layout is implemented");
  }
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  AYAKA_DISPATCH_FLOAT_DTYPES(query.dtype, "paged_attention_v2", [&] {
    return launch_paged_attention_v2<scalar_t>(
        out, query, kv_cache, block_tables, seq_lens, exp_sums, max_logits,
        tmp_out, scale, cuda_stream);
  });
}
}  // namespace ayaka