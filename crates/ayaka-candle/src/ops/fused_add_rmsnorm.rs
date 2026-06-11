//! Fused residual-add + RMSNorm, cloned from the rmsnorm tracer pattern.

use candle_core::{D, DType, Result, Tensor};

/// Pure-candle reference (any device, f32 accumulation). Returns
/// `(normed, input + residual)` like the native kernel.
pub fn fused_add_rmsnorm_ref(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let x = (input.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?;
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps as f64)?.sqrt()?;
    let normed = x
        .broadcast_div(&rms)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    Ok((normed.to_dtype(input.dtype())?, x.to_dtype(input.dtype())?))
}

/// Allocate fresh outputs and run the native kernel. Returns
/// `(normed, input + residual)`.
#[cfg(feature = "cuda")]
pub fn fused_add_rmsnorm_new(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let out = input.zeros_like()?;
    let residual_out = input.zeros_like()?;
    fused_add_rmsnorm(&out, &residual_out, input, residual, weight, eps)?;
    Ok((out, residual_out))
}

#[cfg(feature = "cuda")]
pub use cuda::fused_add_rmsnorm;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{D, Result, Tensor};

    use crate::extract::{self, dispatch_float_dtype, extract_views};

    /// Run the native fused add+RMSNorm kernel:
    /// `residual_out = input + residual`, `out = rmsnorm(residual_out) * weight`.
    ///
    /// All tensors must be contiguous CUDA tensors of one dtype (f32/f16/bf16)
    /// on the same device. In-place is supported: `out` may alias `input` and
    /// `residual_out` may alias `residual`; `out` must not alias `residual_out`.
    pub fn fused_add_rmsnorm(
        out: &Tensor,
        residual_out: &Tensor,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<()> {
        let dtype = input.dtype();
        for (t, name) in [
            (out, "out"),
            (residual_out, "residual_out"),
            (residual, "residual"),
            (weight, "weight"),
        ] {
            if t.dtype() != dtype {
                candle_core::bail!(
                    "fused_add_rmsnorm: {name} dtype {:?} must match input {:?}",
                    t.dtype(),
                    dtype
                );
            }
            if t.device().location() != input.device().location() {
                candle_core::bail!("fused_add_rmsnorm: {name} must be on input's device");
            }
        }
        for (t, name) in [
            (out, "out"),
            (residual_out, "residual_out"),
            (residual, "residual"),
        ] {
            if t.dims() != input.dims() {
                candle_core::bail!(
                    "fused_add_rmsnorm: {name} shape {:?} must match input shape {:?}",
                    t.dims(),
                    input.dims()
                );
            }
        }
        if weight.rank() != 1 || weight.dim(0)? != input.dim(D::Minus1)? {
            candle_core::bail!(
                "fused_add_rmsnorm: weight shape {:?} must be [input last dim {}]",
                weight.dims(),
                input.dim(D::Minus1)?
            );
        }

        dispatch_float_dtype!(dtype, "fused_add_rmsnorm",
            T => launch::<T>(out, residual_out, input, residual, weight, eps))
    }

    fn launch<T: CudaDType>(
        out: &Tensor,
        residual_out: &Tensor,
        input: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<()> {
        let device = input.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(input.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "fused_add_rmsnorm out"),
            res_out_view <- (residual_out, T, dtype, "fused_add_rmsnorm residual_out"),
            in_view <- (input, T, dtype, "fused_add_rmsnorm input"),
            res_view <- (residual, T, dtype, "fused_add_rmsnorm residual"),
            w_view <- (weight, T, dtype, "fused_add_rmsnorm weight"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::fused_add_rmsnorm(
                &out_view,
                &res_out_view,
                &in_view,
                &res_view,
                &w_view,
                eps,
                raw_stream,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
