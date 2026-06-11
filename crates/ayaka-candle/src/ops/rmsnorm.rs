//! RMSNorm — the tracer-bullet op exercising the full ayaka stack:
//! Tensor → extract → kernel-api → C ABI → CUDA kernel.

use candle_core::{D, DType, Result, Tensor};

/// Pure-candle reference implementation (any device, f32 accumulation).
///
/// Serves as the numeric oracle in tests and as the fallback path while a
/// device has no native kernel.
pub fn rmsnorm_ref(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let x = input.to_dtype(DType::F32)?;
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps as f64)?.sqrt()?;
    x.broadcast_div(&rms)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())
}

/// Allocate a fresh output tensor and run the native RMSNorm kernel into it.
#[cfg(feature = "cuda")]
pub fn rmsnorm_new(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let out = input.zeros_like()?;
    rmsnorm(&out, input, weight, eps)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub use cuda::rmsnorm;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{D, DType, Result, Tensor};
    use half::{bf16, f16};

    use crate::extract;

    /// Run the native RMSNorm kernel: `out = input * rsqrt(mean(input², -1) + eps) * weight`.
    ///
    /// All tensors must be contiguous CUDA tensors of one dtype (f32/f16/bf16)
    /// on the same device. `out` must not share storage with any tensor other
    /// than `input` (writing in place over `input` is supported).
    pub fn rmsnorm(
        out: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<()> {
        let dtype = input.dtype();
        if out.dtype() != dtype || weight.dtype() != dtype {
            candle_core::bail!(
                "rmsnorm: dtype mismatch (out {:?}, input {:?}, weight {:?})",
                out.dtype(),
                dtype,
                weight.dtype()
            );
        }
        let loc = input.device().location();
        if out.device().location() != loc || weight.device().location() != loc {
            candle_core::bail!("rmsnorm: out, input, and weight must be on the same device");
        }
        if out.dims() != input.dims() {
            candle_core::bail!(
                "rmsnorm: out shape {:?} must match input shape {:?}",
                out.dims(),
                input.dims()
            );
        }
        if weight.rank() != 1 {
            candle_core::bail!("rmsnorm: weight must be rank 1, got {:?}", weight.dims());
        }
        if weight.dim(0)? != input.dim(D::Minus1)? {
            candle_core::bail!(
                "rmsnorm: weight length {} must match input last dim {}",
                weight.dim(0)?,
                input.dim(D::Minus1)?
            );
        }

        match dtype {
            DType::F32 => launch::<f32>(out, input, weight, eps),
            DType::F16 => launch::<f16>(out, input, weight, eps),
            DType::BF16 => launch::<bf16>(out, input, weight, eps),
            dt => candle_core::bail!("rmsnorm: unsupported dtype {dt:?}"),
        }
    }

    fn launch<T: CudaDType>(
        out: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<()> {
        let device = input.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(input.dtype())?;

        let (out_storage, out_layout) = out.storage_and_layout();
        let (in_storage, in_layout) = input.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        extract::require_contiguous(out_layout, "rmsnorm out")?;
        extract::require_contiguous(in_layout, "rmsnorm input")?;
        extract::require_contiguous(w_layout, "rmsnorm weight")?;

        let out_slice = extract::cuda_storage(&out_storage, "rmsnorm out")?
            .as_cuda_slice::<T>()?
            .slice(out_layout.start_offset()..);
        let in_slice = extract::cuda_storage(&in_storage, "rmsnorm input")?
            .as_cuda_slice::<T>()?
            .slice(in_layout.start_offset()..);
        let w_slice = extract::cuda_storage(&w_storage, "rmsnorm weight")?
            .as_cuda_slice::<T>()?
            .slice(w_layout.start_offset()..);

        // The guards order this launch after candle's pending work on these
        // buffers; hold them until the kernel is enqueued.
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let (in_ptr, _in_guard) = in_slice.device_ptr(&stream);
        let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);

        let out_view = extract::contiguous_view(out_layout, dtype, ordinal, out_ptr);
        let in_view = extract::contiguous_view(in_layout, dtype, ordinal, in_ptr);
        let w_view = extract::contiguous_view(w_layout, dtype, ordinal, w_ptr);
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: the views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch is enqueued
        // on the same stream candle uses for this device, so the buffers stay
        // valid until the kernel completes.
        unsafe { ayaka_kernel_api::rmsnorm(&out_view, &in_view, &w_view, eps, raw_stream) }
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
