//! SwiGLU (`silu_and_mul`), cloned from the rmsnorm tracer pattern.

use candle_core::{DType, Result, Tensor};

/// Pure-candle reference (any device, f32 accumulation):
/// `out[.., d] = silu(input[.., d]) * input[.., hidden + d]`.
pub fn silu_and_mul_ref(input: &Tensor) -> Result<Tensor> {
    let last = input.rank() - 1;
    let hidden = input.dim(last)? / 2;
    let x = input.to_dtype(DType::F32)?;
    let gate = x.narrow(last, 0, hidden)?;
    let up = x.narrow(last, hidden, hidden)?;
    (gate.silu()? * up)?.to_dtype(input.dtype())
}

/// Allocate a fresh output (last dim halved) and run the native kernel.
#[cfg(feature = "cuda")]
pub fn silu_and_mul_new(input: &Tensor) -> Result<Tensor> {
    let last = input.rank() - 1;
    let in_last = input.dim(last)?;
    if in_last % 2 != 0 {
        candle_core::bail!("silu_and_mul: input last dim {in_last} must be even");
    }
    let mut dims = input.dims().to_vec();
    dims[last] = in_last / 2;
    let out = Tensor::zeros(dims, input.dtype(), input.device())?;
    silu_and_mul(&out, input)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub use cuda::silu_and_mul;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{D, DType, Result, Tensor};
    use half::{bf16, f16};

    use crate::extract;

    /// Run the native SwiGLU kernel into `out`.
    ///
    /// `out` and `input` must be contiguous CUDA tensors of one dtype
    /// (f32/f16/bf16) on the same device, with matching leading dims and
    /// `input.last_dim == 2 * out.last_dim`. `out` must not share storage
    /// with `input`.
    pub fn silu_and_mul(
        out: &Tensor,
        input: &Tensor,
    ) -> Result<()> {
        let dtype = input.dtype();
        if out.dtype() != dtype {
            candle_core::bail!(
                "silu_and_mul: out dtype {:?} must match input {:?}",
                out.dtype(),
                dtype
            );
        }
        if out.device().location() != input.device().location() {
            candle_core::bail!("silu_and_mul: out must be on input's device");
        }
        if out.rank() != input.rank()
            || out.dims()[..out.rank() - 1] != input.dims()[..input.rank() - 1]
        {
            candle_core::bail!(
                "silu_and_mul: out shape {:?} must match input leading dims {:?}",
                out.dims(),
                input.dims()
            );
        }
        if input.dim(D::Minus1)? != 2 * out.dim(D::Minus1)? {
            candle_core::bail!(
                "silu_and_mul: input last dim {} must be 2 * out last dim {}",
                input.dim(D::Minus1)?,
                out.dim(D::Minus1)?
            );
        }

        match dtype {
            DType::F32 => launch::<f32>(out, input),
            DType::F16 => launch::<f16>(out, input),
            DType::BF16 => launch::<bf16>(out, input),
            dt => candle_core::bail!("silu_and_mul: unsupported dtype {dt:?}"),
        }
    }

    fn launch<T: CudaDType>(
        out: &Tensor,
        input: &Tensor,
    ) -> Result<()> {
        let device = input.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(input.dtype())?;

        let (out_storage, out_layout) = out.storage_and_layout();
        let (in_storage, in_layout) = input.storage_and_layout();
        extract::require_contiguous(out_layout, "silu_and_mul out")?;
        extract::require_contiguous(in_layout, "silu_and_mul input")?;

        let out_slice = extract::cuda_storage(&out_storage, "silu_and_mul out")?
            .as_cuda_slice::<T>()?
            .slice(out_layout.start_offset()..);
        let in_slice = extract::cuda_storage(&in_storage, "silu_and_mul input")?
            .as_cuda_slice::<T>()?
            .slice(in_layout.start_offset()..);

        let (out_ptr, _g0) = out_slice.device_ptr(&stream);
        let (in_ptr, _g1) = in_slice.device_ptr(&stream);

        let out_view = extract::contiguous_view(out_layout, dtype, ordinal, out_ptr);
        let in_view = extract::contiguous_view(in_layout, dtype, ordinal, in_ptr);
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe { ayaka_kernel_api::silu_and_mul(&out_view, &in_view, raw_stream) }
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
