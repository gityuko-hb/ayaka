//! KV-cache write path: scatter new key/value tokens into a paged cache.

pub use ayaka_core::layout::KvLayout;
use candle_core::{DType, Result, Tensor};

/// Flat element offset of `(slot, kv, h, d)` in an NHD paged KV cache shaped
/// `[num_blocks, 2, block_size, H_kv, D]` (`kv`: 0 = K, 1 = V). Mirrors
/// `nhd_kv_offset` in `native/csrc/kv_cache/kv_offset.cuh`.
fn nhd_kv_offset(
    slot: i64,
    kv: usize,
    h: usize,
    d: usize,
    block_size: usize,
    h_kv: usize,
    head_dim: usize,
) -> usize {
    let slot = slot as usize;
    let block = slot / block_size;
    let s = slot % block_size;
    (((block * 2 + kv) * block_size + s) * h_kv + h) * head_dim + d
}

/// Pure-candle reference (any device). Returns the updated cache instead of
/// mutating in place, so callers compare against the kernel's read-back.
///
/// `kv_cache` is `[num_blocks, 2, block_size, H_kv, D]`, `key`/`value` are
/// `[N, H_kv, D]` (same dtype as the cache), and `slot_mapping` is `[N]` of
/// flat slot indices (or `-1` to skip). The copy is bit-exact; the f32 round
/// trip is lossless because the inputs are already representable in the cache
/// dtype.
pub fn append_kv_ref(
    kv_cache: &Tensor,
    key: &Tensor,
    value: &Tensor,
    slot_mapping: &[i64],
) -> Result<Tensor> {
    let (_num_blocks, two, block_size, h_kv, head_dim) = kv_cache.dims5()?;
    debug_assert_eq!(two, 2);
    let (n, _, _) = key.dims3()?;
    let orig_dtype = kv_cache.dtype();

    let mut cache: Vec<f32> = kv_cache
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let key_f: Vec<f32> = key
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let value_f: Vec<f32> = value
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    let hd = h_kv * head_dim;
    for (token, &slot) in slot_mapping.iter().enumerate().take(n) {
        if slot < 0 {
            continue;
        }
        for h in 0..h_kv {
            for d in 0..head_dim {
                let src = token * hd + h * head_dim + d;
                cache[nhd_kv_offset(slot, 0, h, d, block_size, h_kv, head_dim)] = key_f[src];
                cache[nhd_kv_offset(slot, 1, h, d, block_size, h_kv, head_dim)] = value_f[src];
            }
        }
    }

    Tensor::from_vec(cache, kv_cache.dims(), kv_cache.device())?.to_dtype(orig_dtype)
}

#[cfg(feature = "cuda")]
pub use cuda::append_kv;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_core::dtype::DType as AyakaDType;
    use ayaka_core::layout::KvLayout;
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};

    use crate::extract::{self, dispatch_float_dtype, extract_views};

    /// Scatter `key`/`value` into `kv_cache` **in place** on the GPU.
    ///
    /// `key`/`value` are `[N, H_kv, D]` contiguous CUDA tensors of one dtype
    /// (f32/f16/bf16); `kv_cache` is `[num_blocks, 2, block_size, H_kv, D]` of
    /// the same dtype; `slot_mapping` is i64 `[N]` on the same device. Each
    /// slot must be in range or `-1` (not validated here). Only NHD layout is
    /// implemented.
    pub fn append_kv(
        key: &Tensor,
        value: &Tensor,
        kv_cache: &Tensor,
        slot_mapping: &Tensor,
        layout: KvLayout,
    ) -> Result<()> {
        let dtype = key.dtype();
        if value.dtype() != dtype {
            candle_core::bail!(
                "append_kv: value dtype {:?} must match key {:?}",
                value.dtype(),
                dtype
            );
        }
        if kv_cache.dtype() != dtype {
            candle_core::bail!(
                "append_kv: kv_cache dtype {:?} must match key {:?}",
                kv_cache.dtype(),
                dtype
            );
        }
        if slot_mapping.dtype() != DType::I64 {
            candle_core::bail!(
                "append_kv: slot_mapping must be i64, got {:?}",
                slot_mapping.dtype()
            );
        }
        let loc = key.device().location();
        for (t, name) in [
            (value, "value"),
            (kv_cache, "kv_cache"),
            (slot_mapping, "slot_mapping"),
        ] {
            if t.device().location() != loc {
                candle_core::bail!("append_kv: {name} must be on key's device");
            }
        }

        let (n, h_kv, head_dim) = key.dims3()?;
        let (v_n, v_h, v_d) = value.dims3()?;
        if (v_n, v_h, v_d) != (n, h_kv, head_dim) {
            candle_core::bail!(
                "append_kv: value shape {:?} must match key {:?}",
                value.dims(),
                key.dims()
            );
        }
        let (_blocks, two, _block_size, c_h, c_d) = kv_cache.dims5()?;
        if two != 2 {
            candle_core::bail!("append_kv: kv_cache dim 1 must be 2, got {two}");
        }
        if c_h != h_kv || c_d != head_dim {
            candle_core::bail!(
                "append_kv: kv_cache shape {:?} must share H_kv/D with key {:?}",
                kv_cache.dims(),
                key.dims()
            );
        }
        if slot_mapping.dims() != [n] {
            candle_core::bail!(
                "append_kv: slot_mapping shape {:?} must be [{n}]",
                slot_mapping.dims()
            );
        }

        dispatch_float_dtype!(dtype, "append_kv",
            T => launch::<T>(key, value, kv_cache, slot_mapping, layout))
    }

    fn launch<T: CudaDType>(
        key: &Tensor,
        value: &Tensor,
        kv_cache: &Tensor,
        slot_mapping: &Tensor,
        layout: KvLayout,
    ) -> Result<()> {
        let device = key.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(key.dtype())?;

        extract_views!(&stream, ordinal;
            k_view <- (key, T, dtype, "append_kv key"),
            v_view <- (value, T, dtype, "append_kv value"),
            c_view <- (kv_cache, T, dtype, "append_kv kv_cache"),
            s_view <- (slot_mapping, i64, AyakaDType::I64, "append_kv slot_mapping"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage guards are held across the call; `kv_cache` is the only
        // tensor written and does not alias the inputs. The launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::append_kv(&k_view, &v_view, &c_view, &s_view, layout, raw_stream)
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
