//! Paged attention, decode phase: one query token per sequence attends over
//! its paged K/V cache (NHD layout). Mirrors the append_kv / gemm op pattern.

pub use ayaka_core::layout::KvLayout;
use candle_core::{DType, Result, Tensor};

/// Flat element offset of `(slot, kv, h, d)` in an NHD paged KV cache shaped
/// `[num_blocks, 2, block_size, H_kv, D]` (`kv`: 0 = K, 1 = V). Mirrors
/// `nhd_kv_offset` in `native/csrc/kv_cache/kv_offset.cuh`.
fn nhd_kv_offset(
    slot: usize,
    kv: usize,
    h: usize,
    d: usize,
    block_size: usize,
    h_kv: usize,
    head_dim: usize,
) -> usize {
    let block = slot / block_size;
    let s = slot % block_size;
    (((block * 2 + kv) * block_size + s) * h_kv + h) * head_dim + d
}

/// Pure-candle reference (any device, f32 math). Returns `[num_seqs,
/// num_heads, head_dim]`.
///
/// `query` is `[num_seqs, num_heads, head_dim]`, `kv_cache` is `[num_blocks, 2,
/// block_size, H_kv, head_dim]`, `block_tables` is a row-major `[num_seqs,
/// max_num_blocks]` slice of physical block ids, and `seq_lens[s]` is the
/// number of cached tokens for sequence `s`.
pub fn paged_attention_decode_ref(
    query: &Tensor,
    kv_cache: &Tensor,
    block_tables: &[i64],
    seq_lens: &[i64],
    scale: f32,
) -> Result<Tensor> {
    let (num_seqs, num_heads, head_dim) = query.dims3()?;
    let (_blocks, two, block_size, h_kv, c_d) = kv_cache.dims5()?;
    debug_assert_eq!(two, 2);
    debug_assert_eq!(c_d, head_dim);
    let group = num_heads / h_kv;
    let max_num_blocks = if num_seqs == 0 {
        0
    } else {
        block_tables.len() / num_seqs
    };

    let q: Vec<f32> = query
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let cache: Vec<f32> = kv_cache
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut out = vec![0f32; num_seqs * num_heads * head_dim];

    for s in 0..num_seqs {
        let len = seq_lens[s].max(0) as usize;
        if len == 0 {
            continue;
        }
        for h in 0..num_heads {
            let kv_head = h / group;
            let q_base = (s * num_heads + h) * head_dim;

            // Logits, then a numerically stable softmax.
            let mut scores = vec![0f32; len];
            let mut max_logit = f32::NEG_INFINITY;
            for (t, score) in scores.iter_mut().enumerate() {
                let blk = block_tables[s * max_num_blocks + t / block_size] as usize;
                let slot = blk * block_size + (t % block_size);
                let mut dot = 0f32;
                for d in 0..head_dim {
                    let koff = nhd_kv_offset(slot, 0, kv_head, d, block_size, h_kv, head_dim);
                    dot += q[q_base + d] * cache[koff];
                }
                *score = dot * scale;
                max_logit = max_logit.max(*score);
            }
            let mut denom = 0f32;
            for score in &mut scores {
                *score = (*score - max_logit).exp();
                denom += *score;
            }

            let o_base = (s * num_heads + h) * head_dim;
            for (t, &score) in scores.iter().enumerate() {
                let p = score / denom;
                let blk = block_tables[s * max_num_blocks + t / block_size] as usize;
                let slot = blk * block_size + (t % block_size);
                for d in 0..head_dim {
                    let voff = nhd_kv_offset(slot, 1, kv_head, d, block_size, h_kv, head_dim);
                    out[o_base + d] += p * cache[voff];
                }
            }
        }
    }

    Tensor::from_vec(out, (num_seqs, num_heads, head_dim), query.device())?.to_dtype(query.dtype())
}

/// Host reference for split-KV (v2): compute attention by partitioning each
/// sequence into `partition_size`-token chunks, taking a local online softmax
/// per chunk, then recombining the chunks with a log-sum-exp — the exact
/// algorithm the v2 kernels implement. For any `partition_size >= 1` this must
/// equal [`paged_attention_decode_ref`]; the matching unit test pins that.
pub fn paged_attention_decode_partitioned_ref(
    query: &Tensor,
    kv_cache: &Tensor,
    block_tables: &[i64],
    seq_lens: &[i64],
    scale: f32,
    partition_size: usize,
) -> Result<Tensor> {
    assert!(partition_size >= 1, "partition_size must be >= 1");
    let (num_seqs, num_heads, head_dim) = query.dims3()?;
    let (_blocks, two, block_size, h_kv, c_d) = kv_cache.dims5()?;
    debug_assert_eq!(two, 2);
    debug_assert_eq!(c_d, head_dim);
    let group = num_heads / h_kv;
    let max_num_blocks = if num_seqs == 0 {
        0
    } else {
        block_tables.len() / num_seqs
    };

    let q: Vec<f32> = query
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let cache: Vec<f32> = kv_cache
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut out = vec![0f32; num_seqs * num_heads * head_dim];

    for s in 0..num_seqs {
        let len = seq_lens[s].max(0) as usize;
        if len == 0 {
            continue;
        }
        let num_parts = len.div_ceil(partition_size);
        for h in 0..num_heads {
            let kv_head = h / group;
            let q_base = (s * num_heads + h) * head_dim;

            // Per-partition partials: (max_logit, exp_sum, un-normalized acc).
            let mut part_max = vec![f32::NEG_INFINITY; num_parts];
            let mut part_sum = vec![0f32; num_parts];
            let mut part_acc = vec![0f32; num_parts * head_dim];

            for (p, (pmax, psum)) in part_max
                .iter_mut()
                .zip(part_sum.iter_mut())
                .enumerate()
            {
                let lo = p * partition_size;
                let hi = ((p + 1) * partition_size).min(len);
                let mut m = f32::NEG_INFINITY;
                let mut l = 0f32;
                for t in lo..hi {
                    let blk = block_tables[s * max_num_blocks + t / block_size] as usize;
                    let slot = blk * block_size + (t % block_size);
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        let koff = nhd_kv_offset(slot, 0, kv_head, d, block_size, h_kv, head_dim);
                        dot += q[q_base + d] * cache[koff];
                    }
                    let score = dot * scale;
                    let m_new = m.max(score);
                    let corr = (m - m_new).exp();
                    let pe = (score - m_new).exp();
                    l = l * corr + pe;
                    let acc = &mut part_acc[p * head_dim..(p + 1) * head_dim];
                    for (d, a) in acc.iter_mut().enumerate() {
                        let voff = nhd_kv_offset(slot, 1, kv_head, d, block_size, h_kv, head_dim);
                        *a = *a * corr + pe * cache[voff];
                    }
                    m = m_new;
                }
                *pmax = m;
                *psum = l;
            }

            // Recombine partitions with a log-sum-exp.
            let global_max = part_max
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for (m, l) in part_max.iter().zip(part_sum.iter()) {
                denom += l * (m - global_max).exp();
            }
            let inv = if denom > 0.0 {
                1.0 / denom
            } else {
                0.0
            };
            let o_base = (s * num_heads + h) * head_dim;
            for (p, &m) in part_max.iter().enumerate() {
                let w = (m - global_max).exp();
                let acc = &part_acc[p * head_dim..(p + 1) * head_dim];
                for d in 0..head_dim {
                    out[o_base + d] += w * acc[d] * inv;
                }
            }
        }
    }

    Tensor::from_vec(out, (num_seqs, num_heads, head_dim), query.device())?.to_dtype(query.dtype())
}

#[cfg(feature = "cuda")]
pub use cuda::{
    paged_attention_decode, paged_attention_decode_auto, paged_attention_decode_new,
    paged_attention_decode_v2, paged_attention_partition_size,
};

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_core::dtype::DType as AyakaDType;
    use ayaka_core::layout::KvLayout;
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};

    use crate::extract::{self, dispatch_float_dtype, extract_views};

    /// Allocate the `[num_seqs, num_heads, head_dim]` output and run the
    /// decode kernel.
    pub fn paged_attention_decode_new(
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f32,
        layout: KvLayout,
    ) -> Result<Tensor> {
        let out = Tensor::zeros(query.dims(), query.dtype(), query.device())?;
        paged_attention_decode(&out, query, kv_cache, block_tables, seq_lens, scale, layout)?;
        Ok(out)
    }

    /// Decode paged attention into `out` (same shape/dtype as `query`).
    ///
    /// `query`/`out` are `[num_seqs, num_heads, head_dim]`; `kv_cache` is
    /// `[num_blocks, 2, block_size, H_kv, head_dim]`; `block_tables` is i64
    /// `[num_seqs, max_num_blocks]` and `seq_lens` is i64 `[num_seqs]`, all on
    /// the same CUDA device. `num_heads` must be a multiple of `H_kv`. Only NHD
    /// layout is implemented.

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode(
        out: &Tensor,
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f32,
        layout: KvLayout,
    ) -> Result<()> {
        let dtype = query.dtype();
        if out.dtype() != dtype || kv_cache.dtype() != dtype {
            candle_core::bail!(
                "paged_attention: out {:?} / kv_cache {:?} must match query {:?}",
                out.dtype(),
                kv_cache.dtype(),
                dtype
            );
        }
        if block_tables.dtype() != DType::I64 || seq_lens.dtype() != DType::I64 {
            candle_core::bail!("paged_attention: block_tables and seq_lens must be i64");
        }
        let loc = query.device().location();
        for (t, name) in [
            (out, "out"),
            (kv_cache, "kv_cache"),
            (block_tables, "block_tables"),
            (seq_lens, "seq_lens"),
        ] {
            if t.device().location() != loc {
                candle_core::bail!("paged_attention: {name} must be on query's device");
            }
        }

        let (num_seqs, num_heads, head_dim) = query.dims3()?;
        if out.dims() != [num_seqs, num_heads, head_dim] {
            candle_core::bail!(
                "paged_attention: out shape {:?} must match query {:?}",
                out.dims(),
                query.dims()
            );
        }
        let (_blocks, two, _block_size, h_kv, c_d) = kv_cache.dims5()?;
        if two != 2 {
            candle_core::bail!("paged_attention: kv_cache dim 1 must be 2, got {two}");
        }
        if c_d != head_dim {
            candle_core::bail!(
                "paged_attention: kv_cache head_dim {c_d} must match query {head_dim}"
            );
        }
        if h_kv == 0 || num_heads % h_kv != 0 {
            candle_core::bail!(
                "paged_attention: num_heads {num_heads} must be a multiple of H_kv {h_kv}"
            );
        }
        if block_tables.rank() != 2 || block_tables.dim(0)? != num_seqs {
            candle_core::bail!(
                "paged_attention: block_tables {:?} must be [{num_seqs}, max_blocks]",
                block_tables.dims()
            );
        }
        if seq_lens.dims() != [num_seqs] {
            candle_core::bail!(
                "paged_attention: seq_lens {:?} must be [{num_seqs}]",
                seq_lens.dims()
            );
        }

        dispatch_float_dtype!(dtype, "paged_attention",
            T => launch::<T>(out, query, kv_cache, block_tables, seq_lens, scale, layout))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch<T: CudaDType>(
        out: &Tensor,
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f32,
        layout: KvLayout,
    ) -> Result<()> {
        let device = query.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(query.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "paged_attention out"),
            q_view <- (query, T, dtype, "paged_attention query"),
            c_view <- (kv_cache, T, dtype, "paged_attention kv_cache"),
            bt_view <- (block_tables, i64, AyakaDType::I64, "paged_attention block_tables"),
            sl_view <- (seq_lens, i64, AyakaDType::I64, "paged_attention seq_lens"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: every view describes a live, contiguous CUDA allocation whose
        // storage guards are held across the call; `out` is the only tensor
        // written and does not alias the inputs. The launch goes on candle's
        // stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::paged_attention_decode(
                &out_view, &q_view, &c_view, &bt_view, &sl_view, scale, layout, raw_stream,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    /// Tokens per split-KV partition (native constant).
    pub fn paged_attention_partition_size() -> usize {
        ayaka_kernel_api::paged_attention_partition_size().max(1) as usize
    }

    /// Allocate output + scratch and run the split-KV (v2) decode. `max_seq_len`
    /// sizes the partition scratch (`ceil(max_seq_len / partition_size)`).
    pub fn paged_attention_decode_v2(
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f32,
        layout: KvLayout,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let (num_seqs, num_heads, head_dim) = query.dims3()?;
        let part = paged_attention_partition_size();
        let max_partitions = max_seq_len.div_ceil(part).max(1);
        let dev = query.device();

        let out = Tensor::zeros((num_seqs, num_heads, head_dim), query.dtype(), dev)?;
        let exp_sums = Tensor::zeros((num_seqs, num_heads, max_partitions), DType::F32, dev)?;
        let max_logits = Tensor::zeros((num_seqs, num_heads, max_partitions), DType::F32, dev)?;
        let tmp_out = Tensor::zeros(
            (num_seqs, num_heads, max_partitions, head_dim),
            query.dtype(),
            dev,
        )?;

        dispatch_float_dtype!(query.dtype(), "paged_attention_v2",
            T => launch_v2::<T>(&out, query, kv_cache, block_tables, seq_lens,
                                &exp_sums, &max_logits, &tmp_out, scale, layout))?;
        Ok(out)
    }

    /// Pick v1 (one block per head) for short contexts, v2 (split-KV) once the
    /// longest sequence exceeds one partition — the M6 auto-select.
    pub fn paged_attention_decode_auto(
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f32,
        layout: KvLayout,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        if max_seq_len > paged_attention_partition_size() {
            paged_attention_decode_v2(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                scale,
                layout,
                max_seq_len,
            )
        } else {
            paged_attention_decode_new(query, kv_cache, block_tables, seq_lens, scale, layout)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_v2<T: CudaDType>(
        out: &Tensor,
        query: &Tensor,
        kv_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        exp_sums: &Tensor,
        max_logits: &Tensor,
        tmp_out: &Tensor,
        scale: f32,
        layout: KvLayout,
    ) -> Result<()> {
        let device = query.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(query.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "paged_attention_v2 out"),
            q_view <- (query, T, dtype, "paged_attention_v2 query"),
            c_view <- (kv_cache, T, dtype, "paged_attention_v2 kv_cache"),
            bt_view <- (block_tables, i64, AyakaDType::I64, "paged_attention_v2 block_tables"),
            sl_view <- (seq_lens, i64, AyakaDType::I64, "paged_attention_v2 seq_lens"),
            es_view <- (exp_sums, f32, AyakaDType::F32, "paged_attention_v2 exp_sums"),
            ml_view <- (max_logits, f32, AyakaDType::F32, "paged_attention_v2 max_logits"),
            to_view <- (tmp_out, T, dtype, "paged_attention_v2 tmp_out"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: every view describes a live, contiguous CUDA allocation whose
        // storage guards are held across the call; `out` and the scratch are
        // written and do not alias the inputs. The launch goes on candle's
        // stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::paged_attention_decode_v2(
                &out_view, &q_view, &c_view, &bt_view, &sl_view, &es_view, &ml_view, &to_view,
                scale, layout, raw_stream,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn ref_matches_hand_computed_softmax() {
        // S=1, H=1, H_kv=1, D=2, block_size=2, 1 block, seq_len=2.
        // q=[1,0]; k0=[1,0], k1=[0,1]; v0=[1,0], v1=[0,1]; scale=1.
        // logits=[1,0] -> softmax=[e/(e+1), 1/(e+1)] ~ [0.7311, 0.2689].
        // out = p0*v0 + p1*v1 = [0.7311, 0.2689].
        let cpu = Device::Cpu;
        let q = Tensor::from_vec(vec![1f32, 0.0], (1, 1, 2), &cpu).unwrap();
        // NHD [1,2,2,1,2]: [K_t0(2), K_t1(2), V_t0(2), V_t1(2)]
        let cache = Tensor::from_vec(
            vec![1f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            (1, 2, 2, 1, 2),
            &cpu,
        )
        .unwrap();
        let block_tables = [0i64];
        let seq_lens = [2i64];

        let out = paged_attention_decode_ref(&q, &cache, &block_tables, &seq_lens, 1.0).unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let e = std::f32::consts::E;
        let want = [e / (e + 1.0), 1.0 / (e + 1.0)];
        assert!((got[0] - want[0]).abs() < 1e-5, "got {got:?} want {want:?}");
        assert!((got[1] - want[1]).abs() < 1e-5, "got {got:?} want {want:?}");
    }

    #[test]
    fn ref_zero_seq_len_is_zero() {
        let cpu = Device::Cpu;
        let q = Tensor::from_vec(vec![1f32, 2.0], (1, 1, 2), &cpu).unwrap();
        let cache = Tensor::zeros((1, 2, 2, 1, 2), DType::F32, &cpu).unwrap();
        let out = paged_attention_decode_ref(&q, &cache, &[0i64], &[0i64], 1.0).unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![0.0, 0.0]);
    }

    #[test]
    fn partitioned_ref_matches_monolithic() {
        // A sequence long enough to span several partitions; the split-KV
        // log-sum-exp recombination must reproduce the single-pass softmax.
        let cpu = Device::Cpu;
        let (num_blocks, block_size, h, h_kv, d) = (4, 4, 2, 2, 8);
        let seq_lens = [10i64]; // 3 partitions at partition_size = 4
        let scale = 1.0f32 / (d as f32).sqrt();

        let q = Tensor::randn(0f32, 1f32, (1, h, d), &cpu).unwrap();
        let cache = Tensor::randn(0f32, 1f32, (num_blocks, 2, block_size, h_kv, d), &cpu).unwrap();
        // logical blocks 0..ceil(10/4)=3 -> physical blocks 0,1,2
        let block_tables = [0i64, 1, 2];

        let mono = paged_attention_decode_ref(&q, &cache, &block_tables, &seq_lens, scale).unwrap();
        let part =
            paged_attention_decode_partitioned_ref(&q, &cache, &block_tables, &seq_lens, scale, 4)
                .unwrap();

        let a: Vec<f32> = mono.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = part.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "split-KV recombination diverged: {max_diff}"
        );
    }

    #[test]
    fn partitioned_ref_single_partition_equals_monolithic() {
        // partition_size larger than seq_len -> exactly one partition.
        let cpu = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (2, 2, 8), &cpu).unwrap();
        let cache = Tensor::randn(0f32, 1f32, (4, 2, 4, 2, 8), &cpu).unwrap();
        let block_tables = [0i64, 1, 2, 3]; // [2 seqs, 2 blocks]
        let seq_lens = [3i64, 5];
        let scale = 0.3;
        let mono = paged_attention_decode_ref(&q, &cache, &block_tables, &seq_lens, scale).unwrap();
        let part =
            paged_attention_decode_partitioned_ref(&q, &cache, &block_tables, &seq_lens, scale, 64)
                .unwrap();
        let a: Vec<f32> = mono.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = part.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "single-partition mismatch: {max_diff}");
    }
}
