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
    use candle_core::{D, DType, Result, Tensor};
    use half::{bf16, f16};

    use crate::extract;

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

        match dtype {
            DType::F32 => launch::<f32>(out, residual_out, input, residual, weight, eps),
            DType::F16 => launch::<f16>(out, residual_out, input, residual, weight, eps),
            DType::BF16 => launch::<bf16>(out, residual_out, input, residual, weight, eps),
            dt => candle_core::bail!("fused_add_rmsnorm: unsupported dtype {dt:?}"),
        }
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

        let (out_storage, out_layout) = out.storage_and_layout();
        let (res_out_storage, res_out_layout) = residual_out.storage_and_layout();
        let (in_storage, in_layout) = input.storage_and_layout();
        let (res_storage, res_layout) = residual.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        extract::require_contiguous(out_layout, "fused_add_rmsnorm out")?;
        extract::require_contiguous(res_out_layout, "fused_add_rmsnorm residual_out")?;
        extract::require_contiguous(in_layout, "fused_add_rmsnorm input")?;
        extract::require_contiguous(res_layout, "fused_add_rmsnorm residual")?;
        extract::require_contiguous(w_layout, "fused_add_rmsnorm weight")?;

        let out_slice = extract::cuda_storage(&out_storage, "fused_add_rmsnorm out")?
            .as_cuda_slice::<T>()?
            .slice(out_layout.start_offset()..);
        let res_out_slice =
            extract::cuda_storage(&res_out_storage, "fused_add_rmsnorm residual_out")?
                .as_cuda_slice::<T>()?
                .slice(res_out_layout.start_offset()..);
        let in_slice = extract::cuda_storage(&in_storage, "fused_add_rmsnorm input")?
            .as_cuda_slice::<T>()?
            .slice(in_layout.start_offset()..);
        let res_slice = extract::cuda_storage(&res_storage, "fused_add_rmsnorm residual")?
            .as_cuda_slice::<T>()?
            .slice(res_layout.start_offset()..);
        let w_slice = extract::cuda_storage(&w_storage, "fused_add_rmsnorm weight")?
            .as_cuda_slice::<T>()?
            .slice(w_layout.start_offset()..);

        let (out_ptr, _g0) = out_slice.device_ptr(&stream);
        let (res_out_ptr, _g1) = res_out_slice.device_ptr(&stream);
        let (in_ptr, _g2) = in_slice.device_ptr(&stream);
        let (res_ptr, _g3) = res_slice.device_ptr(&stream);
        let (w_ptr, _g4) = w_slice.device_ptr(&stream);

        let out_view = extract::contiguous_view(out_layout, dtype, ordinal, out_ptr);
        let res_out_view = extract::contiguous_view(res_out_layout, dtype, ordinal, res_out_ptr);
        let in_view = extract::contiguous_view(in_layout, dtype, ordinal, in_ptr);
        let res_view = extract::contiguous_view(res_layout, dtype, ordinal, res_ptr);
        let w_view = extract::contiguous_view(w_layout, dtype, ordinal, w_ptr);
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
