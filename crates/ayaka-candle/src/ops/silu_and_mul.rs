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
    use candle_core::{D, Result, Tensor};

    use crate::extract::{self, dispatch_float_dtype, extract_views};

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

        dispatch_float_dtype!(dtype, "silu_and_mul", T => launch::<T>(out, input))
    }

    fn launch<T: CudaDType>(
        out: &Tensor,
        input: &Tensor,
    ) -> Result<()> {
        let device = input.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(input.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "silu_and_mul out"),
            in_view <- (input, T, dtype, "silu_and_mul input"),
        );
        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: views describe live, contiguous CUDA allocations whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe { ayaka_kernel_api::silu_and_mul(&out_view, &in_view, raw_stream) }
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
