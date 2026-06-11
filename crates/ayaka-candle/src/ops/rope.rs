//! Rotary position embedding (neox + gptj layouts), in-place on q and k.

pub use ayaka_kernel_api::RopeLayout;
use candle_core::{DType, Result, Tensor};

/// Pure-candle reference (any device, f32 math). Returns the rotated tensor
/// instead of mutating in place.
///
/// `x` is `[num_tokens, num_heads, head_dim]`, `positions` is `[num_tokens]`
/// (i64), `cos_sin_cache` is `[max_position, rot_dim]` f32 with cosines in
/// the first half of each row and sines in the second.
pub fn rope_ref(
    x: &Tensor,
    positions: &Tensor,
    cos_sin_cache: &Tensor,
    layout: RopeLayout,
) -> Result<Tensor> {
    let (t, h, head_dim) = x.dims3()?;
    let rot_dim = cos_sin_cache.dim(1)?;
    let half = rot_dim / 2;
    let orig_dtype = x.dtype();

    let xf = x.to_dtype(DType::F32)?;
    let cs = cos_sin_cache
        .to_dtype(DType::F32)?
        .index_select(positions, 0)?;
    let cos = cs.narrow(1, 0, half)?.reshape((t, 1, half))?;
    let sin = cs.narrow(1, half, half)?.reshape((t, 1, half))?;

    let x_rot = xf.narrow(2, 0, rot_dim)?;
    let (x1, x2) = match layout {
        RopeLayout::Neox => (x_rot.narrow(2, 0, half)?, x_rot.narrow(2, half, half)?),
        RopeLayout::GptJ => {
            let pairs = x_rot.reshape((t, h, half, 2))?;
            (
                pairs.narrow(3, 0, 1)?.squeeze(3)?,
                pairs.narrow(3, 1, 1)?.squeeze(3)?,
            )
        },
    };
    let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let o2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    let rotated = match layout {
        RopeLayout::Neox => Tensor::cat(&[&o1, &o2], 2)?,
        RopeLayout::GptJ => Tensor::stack(&[&o1, &o2], 3)?.reshape((t, h, rot_dim))?,
    };

    let out = if rot_dim < head_dim {
        let x_pass = xf.narrow(2, rot_dim, head_dim - rot_dim)?;
        Tensor::cat(&[&rotated, &x_pass], 2)?
    } else {
        rotated
    };
    out.to_dtype(orig_dtype)
}

/// Build a `[max_position, rot_dim]` f32 cos/sin cache on `device` using the
/// standard `theta^(-2i/rot_dim)` frequencies (cosines first, then sines).
pub fn cos_sin_cache(
    max_position: usize,
    rot_dim: usize,
    theta: f32,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let half = rot_dim / 2;
    let mut data = Vec::with_capacity(max_position * rot_dim);
    for pos in 0..max_position {
        for i in 0..rot_dim {
            let d = i % half;
            let freq = theta.powf(-2.0 * d as f32 / rot_dim as f32);
            let angle = pos as f32 * freq;
            data.push(if i < half {
                angle.cos()
            } else {
                angle.sin()
            });
        }
    }
    Tensor::from_vec(data, (max_position, rot_dim), device)
}

#[cfg(feature = "cuda")]
pub use cuda::rope;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_kernel_api::{AyakaStream, RopeLayout};
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};
    use half::{bf16, f16};

    use crate::extract;

    /// Apply rotary embeddings to `query` and `key` **in place**.
    ///
    /// `query` is `[num_tokens, num_heads, head_dim]` and `key` is
    /// `[num_tokens, num_kv_heads, head_dim]`, contiguous CUDA tensors of one
    /// dtype (f32/f16/bf16); `positions` is i64 `[num_tokens]` and
    /// `cos_sin_cache` is f32 `[max_position, rot_dim]` on the same device.
    /// Every position value must be a valid cache row (not validated here).
    /// `query`/`key` must not share storage with other live tensors.
    pub fn rope(
        query: &Tensor,
        key: &Tensor,
        positions: &Tensor,
        cos_sin_cache: &Tensor,
        layout: RopeLayout,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.dtype() != dtype {
            candle_core::bail!(
                "rope: key dtype {:?} must match query {:?}",
                key.dtype(),
                dtype
            );
        }
        if positions.dtype() != DType::I64 {
            candle_core::bail!("rope: positions must be i64, got {:?}", positions.dtype());
        }
        if cos_sin_cache.dtype() != DType::F32 {
            candle_core::bail!(
                "rope: cos_sin_cache must be f32, got {:?}",
                cos_sin_cache.dtype()
            );
        }
        let loc = query.device().location();
        for (t, name) in [
            (key, "key"),
            (positions, "positions"),
            (cos_sin_cache, "cos_sin_cache"),
        ] {
            if t.device().location() != loc {
                candle_core::bail!("rope: {name} must be on query's device");
            }
        }

        let (tokens, _heads, head_dim) = query.dims3()?;
        let (k_tokens, _kv_heads, k_head_dim) = key.dims3()?;
        if k_tokens != tokens || k_head_dim != head_dim {
            candle_core::bail!(
                "rope: key shape {:?} must share tokens/head_dim with query {:?}",
                key.dims(),
                query.dims()
            );
        }
        if positions.dims() != [tokens] {
            candle_core::bail!(
                "rope: positions shape {:?} must be [{tokens}]",
                positions.dims()
            );
        }
        let rot_dim = cos_sin_cache.dim(1)?;
        if rot_dim % 2 != 0 || rot_dim > head_dim {
            candle_core::bail!("rope: rot_dim {rot_dim} must be even and <= head_dim {head_dim}");
        }

        match dtype {
            DType::F32 => launch::<f32>(query, key, positions, cos_sin_cache, layout),
            DType::F16 => launch::<f16>(query, key, positions, cos_sin_cache, layout),
            DType::BF16 => launch::<bf16>(query, key, positions, cos_sin_cache, layout),
            dt => candle_core::bail!("rope: unsupported dtype {dt:?}"),
        }
    }

    fn launch<T: CudaDType>(
        query: &Tensor,
        key: &Tensor,
        positions: &Tensor,
        cos_sin_cache: &Tensor,
        layout: RopeLayout,
    ) -> Result<()> {
        let device = query.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(query.dtype())?;

        let (q_storage, q_layout) = query.storage_and_layout();
        let (k_storage, k_layout) = key.storage_and_layout();
        let (p_storage, p_layout) = positions.storage_and_layout();
        let (c_storage, c_layout) = cos_sin_cache.storage_and_layout();
        extract::require_contiguous(q_layout, "rope query")?;
        extract::require_contiguous(k_layout, "rope key")?;
        extract::require_contiguous(p_layout, "rope positions")?;
        extract::require_contiguous(c_layout, "rope cos_sin_cache")?;

        let q_slice = extract::cuda_storage(&q_storage, "rope query")?
            .as_cuda_slice::<T>()?
            .slice(q_layout.start_offset()..);
        let k_slice = extract::cuda_storage(&k_storage, "rope key")?
            .as_cuda_slice::<T>()?
            .slice(k_layout.start_offset()..);
        let p_slice = extract::cuda_storage(&p_storage, "rope positions")?
            .as_cuda_slice::<i64>()?
            .slice(p_layout.start_offset()..);
        let c_slice = extract::cuda_storage(&c_storage, "rope cos_sin_cache")?
            .as_cuda_slice::<f32>()?
            .slice(c_layout.start_offset()..);

        let (q_ptr, _g0) = q_slice.device_ptr(&stream);
        let (k_ptr, _g1) = k_slice.device_ptr(&stream);
        let (p_ptr, _g2) = p_slice.device_ptr(&stream);
        let (c_ptr, _g3) = c_slice.device_ptr(&stream);

        let q_view = extract::contiguous_view(q_layout, dtype, ordinal, q_ptr);
        let k_view = extract::contiguous_view(k_layout, dtype, ordinal, k_ptr);
        let p_view =
            extract::contiguous_view(p_layout, ayaka_core::dtype::DType::I64, ordinal, p_ptr);
        let c_view =
            extract::contiguous_view(c_layout, ayaka_core::dtype::DType::F32, ordinal, c_ptr);
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe { ayaka_kernel_api::rope(&q_view, &k_view, &p_view, &c_view, layout, raw_stream) }
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
