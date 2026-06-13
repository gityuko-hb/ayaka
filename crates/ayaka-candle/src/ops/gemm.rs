//! Dense GEMM (cuBLASLt wrap), cloned from the rmsnorm tracer pattern.

use candle_core::{DType, Result, Tensor};

/// Pure-candle reference with f64 accumulation:
/// `out = op_a(a) @ op_b(b) (+ bias)`, `op_x` transposing when `trans_x`.
pub fn gemm_ref(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tensor> {
    let a64 = a.to_dtype(DType::F64)?;
    let b64 = b.to_dtype(DType::F64)?;
    let a64 = if trans_a {
        a64.t()?.contiguous()?
    } else {
        a64
    };
    let b64 = if trans_b {
        b64.t()?.contiguous()?
    } else {
        b64
    };
    let mut out = a64.matmul(&b64)?;
    if let Some(bias) = bias {
        out = out.broadcast_add(&bias.to_dtype(DType::F64)?)?;
    }
    out.to_dtype(a.dtype())
}

/// Allocate the [M, N] output and run the native kernel.
#[cfg(feature = "cuda")]
pub fn gemm_new(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    trans_a: bool,
    trans_b: bool,
    workspace: Option<&Tensor>,
) -> Result<Tensor> {
    if a.rank() != 2 || b.rank() != 2 {
        candle_core::bail!(
            "gemm: a and b must be rank 2, got {} and {}",
            a.rank(),
            b.rank()
        );
    }
    let m = if trans_a {
        a.dim(1)?
    } else {
        a.dim(0)?
    };
    let n = if trans_b {
        b.dim(0)?
    } else {
        b.dim(1)?
    };
    let out = Tensor::zeros((m, n), a.dtype(), a.device())?;
    gemm(&out, a, b, bias, trans_a, trans_b, workspace)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub use cuda::gemm;

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};
    use core::ffi::c_void;

    use crate::extract::{self, dispatch_float_dtype, extract_views};

    /// Run the native cuBLASLt GEMM into `out = op_a(a) @ op_b(b) (+ bias)`.
    ///
    /// `out [M,N]`, `a` (stored `[M,K]`, or `[K,M]` when `trans_a`), `b`
    /// (stored `[K,N]`, or `[N,K]` when `trans_b`) and `bias [N]` (when
    /// `Some`) must be contiguous CUDA tensors of one dtype (f32/f16/bf16)
    /// on the same device. `workspace`, when `Some`, is a contiguous u8 CUDA
    /// tensor handed to cuBLASLt for scratch. `out` must not share storage
    /// with any input.
    pub fn gemm(
        out: &Tensor,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        trans_a: bool,
        trans_b: bool,
        workspace: Option<&Tensor>,
    ) -> Result<()> {
        let dtype = a.dtype();
        if out.dtype() != dtype || b.dtype() != dtype {
            candle_core::bail!(
                "gemm: out {:?}, a {:?}, and b {:?} dtype must match",
                out.dtype(),
                dtype,
                b.dtype()
            );
        }
        if out.device().location() != a.device().location()
            || b.device().location() != a.device().location()
        {
            candle_core::bail!("gemm: out, a, and b must be on one device");
        }
        if out.rank() != 2 || a.rank() != 2 || b.rank() != 2 {
            candle_core::bail!("gemm: out, a, and b must be rank 2");
        }
        let m = if trans_a {
            a.dim(1)?
        } else {
            a.dim(0)?
        };
        let k = if trans_a {
            a.dim(0)?
        } else {
            a.dim(1)?
        };
        let k_b = if trans_b {
            b.dim(1)?
        } else {
            b.dim(0)?
        };
        let n = if trans_b {
            b.dim(0)?
        } else {
            b.dim(1)?
        };
        if k != k_b {
            candle_core::bail!(
                "gemm: inner dims must match, op_a(a) has K={k}, op_b(b) has K={k_b}"
            );
        }
        if out.dims() != [m, n] {
            candle_core::bail!("gemm: out shape {:?} must be [{m}, {n}]", out.dims());
        }
        if let Some(bias) = bias {
            if bias.dtype() != dtype {
                candle_core::bail!(
                    "gemm: bias dtype {:?} must match a {:?}",
                    bias.dtype(),
                    dtype
                );
            }
            if bias.device().location() != a.device().location() {
                candle_core::bail!("gemm: bias must be on a's device");
            }
            if bias.dims() != [n] {
                candle_core::bail!("gemm: bias shape {:?} must be [{n}]", bias.dims());
            }
        }
        if let Some(workspace) = workspace {
            if workspace.dtype() != DType::U8 {
                candle_core::bail!("gemm: workspace dtype {:?} must be u8", workspace.dtype());
            }
            if workspace.device().location() != a.device().location() {
                candle_core::bail!("gemm: workspace must be on a's device");
            }
        }

        dispatch_float_dtype!(dtype, "gemm", T => launch::<T>(out, a, b, bias, trans_a, trans_b, workspace))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch<T: CudaDType>(
        out: &Tensor,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        trans_a: bool,
        trans_b: bool,
        workspace: Option<&Tensor>,
    ) -> Result<()> {
        let device = a.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(a.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "gemm out"),
            a_view <- (a, T, dtype, "gemm a"),
            b_view <- (b, T, dtype, "gemm b"),
        );

        // Optional views need the manual version of extract_views! so the
        // storage guards, slices, and sync guards all outlive the call.
        let bias_bind = bias.map(|t| t.storage_and_layout());
        let bias_slice;
        let _bias_guard;
        let bias_view = if let Some((storage, layout)) = &bias_bind {
            extract::require_contiguous(layout, "gemm bias")?;
            bias_slice = extract::cuda_storage(storage, "gemm bias")?
                .as_cuda_slice::<T>()?
                .slice(layout.start_offset()..);
            let (ptr, guard) = bias_slice.device_ptr(&stream);
            _bias_guard = Some(guard);
            Some(extract::contiguous_view(layout, dtype, ordinal, ptr))
        } else {
            _bias_guard = None;
            None
        };

        let ws_bind = workspace.map(|t| (t.storage_and_layout(), t.elem_count()));
        let ws_slice;
        let _ws_guard;
        let (ws_ptr, ws_bytes) = if let Some(((storage, layout), bytes)) = &ws_bind {
            extract::require_contiguous(layout, "gemm workspace")?;
            ws_slice = extract::cuda_storage(storage, "gemm workspace")?
                .as_cuda_slice::<u8>()?
                .slice(layout.start_offset()..);
            let (ptr, guard) = ws_slice.device_ptr(&stream);
            _ws_guard = Some(guard);
            (ptr as usize as *mut c_void, *bytes)
        } else {
            _ws_guard = None;
            (core::ptr::null_mut(), 0)
        };

        let raw_stream: AyakaStream = stream.cu_stream().cast();

        // SAFETY: every view (including the optional bias and workspace)
        // describes a live, contiguous CUDA allocation whose storage
        // read-guards are held across the call; the launch goes on candle's
        // stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::gemm(
                &out_view,
                &a_view,
                &b_view,
                bias_view.as_ref(),
                trans_a,
                trans_b,
                ws_ptr,
                ws_bytes,
                raw_stream,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
