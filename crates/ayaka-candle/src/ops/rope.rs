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
    use ayaka_core::dtype::DType as AyakaDType;
    use ayaka_kernel_api::{AyakaStream, RopeLayout};
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};

    use crate::extract::{self, dispatch_float_dtype, extract_views};

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

        dispatch_float_dtype!(dtype, "rope",
            T => launch::<T>(query, key, positions, cos_sin_cache, layout))
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

        extract_views!(&stream, ordinal;
            q_view <- (query, T, dtype, "rope query"),
            k_view <- (key, T, dtype, "rope key"),
            p_view <- (positions, i64, AyakaDType::I64, "rope positions"),
            c_view <- (cos_sin_cache, f32, AyakaDType::F32, "rope cos_sin_cache"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe { ayaka_kernel_api::rope(&q_view, &k_view, &p_view, &c_view, layout, raw_stream) }
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
