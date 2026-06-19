//! Quantized GEMM (W4A16): pure-candle reference dequant+matmul plus a CUDA
//! FFI dispatch that calls `ayaka_kernel_api::quant_gemm`, and a `QuantLinear`
//! layer that holds repacked 4-bit weights as GPU tensors.
//!
//! `b_quant` is `[N, K/2]` U8 (two 4-bit nibbles per byte: low nibble = even
//! index, high nibble = odd index), `b_scales`/`b_mins` are
//! `[N, K/group_size]` F16. The kernel dequantizes weights on-the-fly using
//! `scale * nibble + min` (when `mins` is present) or `scale * (nibble - 8)`
//! (symmetric, Q4_0 style).

use candle_core::{Device, Result, Tensor};

/// Dequantize packed 4-bit weights to a `[N, K]` f32 tensor.
///
/// `b_quant` is `[N, K/2]` U8, `b_scales`/`b_mins` are `[N, K/group_size]` F16.
/// When `b_mins` is `Some`, weight = `scale * nibble + min`; otherwise the
/// symmetric Q4_0 form `scale * (nibble - 8)` is used.
fn dequantize_weights_ref(
    b_quant: &Tensor,
    b_scales: &Tensor,
    b_mins: Option<&Tensor>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Tensor> {
    let b_quant_vec: Vec<u8> = b_quant.flatten_all()?.to_vec1()?;
    let b_scales_vec: Vec<half::f16> = b_scales.flatten_all()?.to_vec1()?;
    let b_mins_vec: Option<Vec<half::f16>> = if let Some(m) = b_mins {
        Some(m.flatten_all()?.to_vec1()?)
    } else {
        None
    };

    let n_groups = in_features / group_size;
    let mut weights = vec![0f32; out_features * in_features];

    for n in 0..out_features {
        for k in 0..in_features {
            let byte_idx = k / 2;
            let packed = b_quant_vec[n * (in_features / 2) + byte_idx];
            let nibble = if k % 2 == 0 {
                (packed & 0x0F) as i32
            } else {
                ((packed >> 4) & 0x0F) as i32
            };

            let group_idx = k / group_size;
            let scale = b_scales_vec[n * n_groups + group_idx].to_f32();

            let weight = if let Some(ref mins) = b_mins_vec {
                let min = mins[n * n_groups + group_idx].to_f32();
                scale * nibble as f32 + min
            } else {
                scale * (nibble - 8) as f32
            };

            weights[n * in_features + k] = weight;
        }
    }

    Tensor::from_vec(weights, (out_features, in_features), b_quant.device())
}

/// Pure-candle reference for W4A16 GEMM: dequantizes `b_quant` using
/// `b_scales` and optional `b_mins`, then computes `a @ dequant_b.T() (+ bias)`.
///
/// `a` is `[M, K]` F16/BF16/F32, `b_quant` is `[N, K/2]` U8, `b_scales` is
/// `[N, K/group_size]` F16, `b_mins` is `[N, K/group_size]` F16 or None.
/// Returns `[M, N]` in `a`'s dtype.
pub fn quant_gemm_ref(
    a: &Tensor,
    b_quant: &Tensor,
    b_scales: &Tensor,
    b_mins: Option<&Tensor>,
    bias: Option<&Tensor>,
    group_size: usize,
) -> Result<Tensor> {
    if b_quant.dtype() != candle_core::DType::U8 {
        candle_core::bail!(
            "quant_gemm_ref: b_quant dtype {:?} must be u8",
            b_quant.dtype()
        );
    }
    let (_m, k) = a.dims2()?;
    let (n, k_half) = b_quant.dims2()?;
    if k_half * 2 != k {
        candle_core::bail!("quant_gemm_ref: b_quant K/2={k_half} but a K={k}");
    }
    if k % group_size != 0 {
        candle_core::bail!("quant_gemm_ref: K={k} not divisible by group_size={group_size}");
    }
    let n_groups = k / group_size;
    let b_scales_dims = b_scales.dims2()?;
    if b_scales_dims != (n, n_groups) {
        candle_core::bail!(
            "quant_gemm_ref: b_scales {:?} must be [{n}, {n_groups}]",
            b_scales_dims
        );
    }
    if let Some(bm) = b_mins
        && bm.dims2()? != (n, n_groups)
    {
        candle_core::bail!("quant_gemm_ref: b_mins must be [{n}, {n_groups}]");
    }
    if let Some(bias) = bias
        && bias.dims1()? != n
    {
        candle_core::bail!("quant_gemm_ref: bias len must be {n}");
    }

    let b_quant_f32 = dequantize_weights_ref(b_quant, b_scales, b_mins, n, k, group_size)?;
    let b = b_quant_f32.to_dtype(a.dtype())?;
    let mut out = a.matmul(&b.t()?)?;
    if let Some(bias) = bias {
        out = out.broadcast_add(&bias.to_dtype(a.dtype())?)?;
    }
    Ok(out)
}

/// Quantized linear layer: stores weights as packed 4-bit quants + scales,
/// dequantizes on-the-fly during GEMM via the W4A16 kernel.
pub struct QuantLinear {
    /// Packed 4-bit quants `[out_features, in_features/2]` as U8 tensor.
    qweight: Tensor,
    /// Per-group scales `[out_features, in_features/group_size]` as F16 tensor.
    scales: Tensor,
    /// Per-group mins `[out_features, in_features/group_size]` as F16 tensor.
    /// None for schemes without mins (e.g. Q4_0).
    mins: Option<Tensor>,
    /// Optional bias `[out_features]`.
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    group_size: usize,
}

impl QuantLinear {
    /// Build from `RepackedWeights` (CPU bytes) + optional bias, uploading to
    /// `device`.
    pub fn from_repacked(
        rw: &ayaka_quant::RepackedWeights,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        if !rw.in_features.is_multiple_of(rw.group_size) {
            candle_core::bail!(
                "QuantLinear::from_repacked: in_features={} not divisible by group_size={}",
                rw.in_features,
                rw.group_size
            );
        }
        let qweight =
            Tensor::from_vec(rw.qs.clone(), (rw.out_features, rw.in_features / 2), device)?;
        let scales = Tensor::from_vec(
            rw.scales.clone(),
            (rw.out_features, rw.in_features / rw.group_size),
            device,
        )?;
        let mins = if rw.mins.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(
                rw.mins.clone(),
                (rw.out_features, rw.in_features / rw.group_size),
                device,
            )?)
        };
        Ok(Self {
            qweight,
            scales,
            mins,
            bias,
            in_features: rw.in_features,
            out_features: rw.out_features,
            group_size: rw.group_size,
        })
    }

    /// Forward pass: input `[batch, seq, in_features]` -> output
    /// `[batch, seq, out_features]`.
    pub fn forward(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _k) = x.dims3()?;
        let a = x.reshape((b * s, self.in_features))?;
        #[cfg(feature = "cuda")]
        {
            let out = quant_gemm_new(
                &a,
                &self.qweight,
                &self.scales,
                self.mins.as_ref(),
                self.bias.as_ref(),
                self.group_size,
                None,
            )?;
            return out.reshape((b, s, self.out_features));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let out = quant_gemm_ref(
                &a,
                &self.qweight,
                &self.scales,
                self.mins.as_ref(),
                self.bias.as_ref(),
                self.group_size,
            )?;
            out.reshape((b, s, self.out_features))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::quant_gemm;

/// Allocate the `[M, N]` output and run the native W4A16 kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm_new(
    a: &Tensor,
    b_quant: &Tensor,
    b_scales: &Tensor,
    b_mins: Option<&Tensor>,
    bias: Option<&Tensor>,
    group_size: usize,
    workspace: Option<&Tensor>,
) -> Result<Tensor> {
    let (m, _k) = a.dims2()?;
    let (n, _k_half) = b_quant.dims2()?;
    let out = Tensor::zeros((m, n), a.dtype(), a.device())?;
    quant_gemm(
        &out, a, b_quant, b_scales, b_mins, bias, group_size, workspace,
    )?;
    Ok(out)
}

#[cfg(feature = "cuda")]
mod cuda {
    use ayaka_core::dtype::DType as AyakaDType;
    use ayaka_kernel_api::AyakaStream;
    use candle_core::cuda_backend::CudaDType;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Tensor};
    use core::ffi::c_void;

    use crate::extract::{self, dispatch_float_dtype, extract_views};

    /// Run the native W4A16 GEMM into `out = a @ dequant(b_quant) (+ bias)`.
    ///
    /// `out [M,N]`, `a [M,K]` (F16/BF16), `b_quant [N,K/2]` U8, `b_scales`
    /// `[N,K/group_size]` (F16/BF16), `b_mins` `[N,K/group_size]` optional,
    /// `bias [N]` optional, and `workspace` optional U8 must be contiguous
    /// CUDA tensors on the same device. `out` must not alias any input.
    #[allow(clippy::too_many_arguments)]
    pub fn quant_gemm(
        out: &Tensor,
        a: &Tensor,
        b_quant: &Tensor,
        b_scales: &Tensor,
        b_mins: Option<&Tensor>,
        bias: Option<&Tensor>,
        group_size: usize,
        workspace: Option<&Tensor>,
    ) -> Result<()> {
        let dtype = a.dtype();
        if out.dtype() != dtype {
            candle_core::bail!(
                "quant_gemm: out {:?} and a {:?} dtype must match",
                out.dtype(),
                dtype
            );
        }
        if b_quant.dtype() != DType::U8 {
            candle_core::bail!("quant_gemm: b_quant dtype {:?} must be u8", b_quant.dtype());
        }
        if b_scales.dtype() != dtype {
            candle_core::bail!(
                "quant_gemm: b_scales dtype {:?} must match a {:?}",
                b_scales.dtype(),
                dtype
            );
        }
        if out.device().location() != a.device().location()
            || b_quant.device().location() != a.device().location()
            || b_scales.device().location() != a.device().location()
        {
            candle_core::bail!("quant_gemm: out, a, b_quant, b_scales must be on one device");
        }
        if out.rank() != 2 || a.rank() != 2 || b_quant.rank() != 2 || b_scales.rank() != 2 {
            candle_core::bail!("quant_gemm: out, a, b_quant, b_scales must be rank 2");
        }
        let (m, k) = a.dims2()?;
        let (n, k_half) = b_quant.dims2()?;
        if k_half * 2 != k {
            candle_core::bail!("quant_gemm: b_quant K/2={k_half} but a K={k}");
        }
        if !k.is_multiple_of(group_size) {
            candle_core::bail!("quant_gemm: K={k} not divisible by group_size={group_size}");
        }
        let n_groups = k / group_size;
        if b_scales.dims() != [n, n_groups] {
            candle_core::bail!(
                "quant_gemm: b_scales {:?} must be [{n}, {n_groups}]",
                b_scales.dims()
            );
        }
        if out.dims() != [m, n] {
            candle_core::bail!("quant_gemm: out shape {:?} must be [{m}, {n}]", out.dims());
        }
        if let Some(bm) = b_mins {
            if bm.dtype() != dtype {
                candle_core::bail!(
                    "quant_gemm: b_mins dtype {:?} must match a {:?}",
                    bm.dtype(),
                    dtype
                );
            }
            if bm.device().location() != a.device().location() {
                candle_core::bail!("quant_gemm: b_mins must be on a's device");
            }
            if bm.dims() != [n, n_groups] {
                candle_core::bail!(
                    "quant_gemm: b_mins {:?} must be [{n}, {n_groups}]",
                    bm.dims()
                );
            }
        }
        if let Some(bias) = bias {
            if bias.dtype() != dtype {
                candle_core::bail!(
                    "quant_gemm: bias dtype {:?} must match a {:?}",
                    bias.dtype(),
                    dtype
                );
            }
            if bias.device().location() != a.device().location() {
                candle_core::bail!("quant_gemm: bias must be on a's device");
            }
            if bias.dims() != [n] {
                candle_core::bail!("quant_gemm: bias shape {:?} must be [{n}]", bias.dims());
            }
        }
        if let Some(workspace) = workspace {
            if workspace.dtype() != DType::U8 {
                candle_core::bail!(
                    "quant_gemm: workspace dtype {:?} must be u8",
                    workspace.dtype()
                );
            }
            if workspace.device().location() != a.device().location() {
                candle_core::bail!("quant_gemm: workspace must be on a's device");
            }
        }

        dispatch_float_dtype!(dtype, "quant_gemm", T => launch::<T>(
            out, a, b_quant, b_scales, b_mins, bias, group_size, workspace
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch<T: CudaDType>(
        out: &Tensor,
        a: &Tensor,
        b_quant: &Tensor,
        b_scales: &Tensor,
        b_mins: Option<&Tensor>,
        bias: Option<&Tensor>,
        group_size: usize,
        workspace: Option<&Tensor>,
    ) -> Result<()> {
        let device = a.device();
        let stream = extract::cuda_stream(device)?;
        let ordinal = extract::cuda_ordinal(device)?;
        let dtype = extract::ayaka_dtype(a.dtype())?;

        extract_views!(&stream, ordinal;
            out_view <- (out, T, dtype, "quant_gemm out"),
            a_view <- (a, T, dtype, "quant_gemm a"),
            b_scales_view <- (b_scales, T, dtype, "quant_gemm b_scales"),
        );

        // b_quant is U8, not the float T, so extract it manually with the
        // U8 ayaka dtype (the macro assumes a single typed slice).
        let (bq_storage, bq_layout) = b_quant.storage_and_layout();
        extract::require_contiguous(&bq_layout, "quant_gemm b_quant")?;
        let bq_slice = extract::cuda_storage(&bq_storage, "quant_gemm b_quant")?
            .as_cuda_slice::<u8>()?
            .slice(bq_layout.start_offset()..);
        let (bq_ptr, _bq_guard) = bq_slice.device_ptr(&stream);
        let b_quant_view = extract::contiguous_view(&bq_layout, AyakaDType::U8, ordinal, bq_ptr);

        // Optional b_mins (float T).
        let b_mins_bind = b_mins.map(|t| t.storage_and_layout());
        let b_mins_slice;
        let _b_mins_guard;
        let b_mins_view = if let Some((storage, layout)) = &b_mins_bind {
            extract::require_contiguous(layout, "quant_gemm b_mins")?;
            b_mins_slice = extract::cuda_storage(storage, "quant_gemm b_mins")?
                .as_cuda_slice::<T>()?
                .slice(layout.start_offset()..);
            let (ptr, guard) = b_mins_slice.device_ptr(&stream);
            _b_mins_guard = Some(guard);
            Some(extract::contiguous_view(layout, dtype, ordinal, ptr))
        } else {
            _b_mins_guard = None;
            None
        };

        // Optional bias (float T).
        let bias_bind = bias.map(|t| t.storage_and_layout());
        let bias_slice;
        let _bias_guard;
        let bias_view = if let Some((storage, layout)) = &bias_bind {
            extract::require_contiguous(layout, "quant_gemm bias")?;
            bias_slice = extract::cuda_storage(storage, "quant_gemm bias")?
                .as_cuda_slice::<T>()?
                .slice(layout.start_offset()..);
            let (ptr, guard) = bias_slice.device_ptr(&stream);
            _bias_guard = Some(guard);
            Some(extract::contiguous_view(layout, dtype, ordinal, ptr))
        } else {
            _bias_guard = None;
            None
        };

        // Optional workspace (U8).
        let ws_bind = workspace.map(|t| (t.storage_and_layout(), t.elem_count()));
        let ws_slice;
        let _ws_guard;
        let (ws_ptr, ws_bytes) = if let Some(((storage, layout), bytes)) = &ws_bind {
            extract::require_contiguous(layout, "quant_gemm workspace")?;
            ws_slice = extract::cuda_storage(storage, "quant_gemm workspace")?
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

        // SAFETY: every view (including the optional b_mins, bias, and
        // workspace) describes a live, contiguous CUDA allocation whose
        // storage read-guards are held across the call; the launch goes on
        // candle's stream for this device, ordering it after pending work.
        unsafe {
            ayaka_kernel_api::quant_gemm(
                &out_view,
                &a_view,
                &b_quant_view,
                &b_scales_view,
                b_mins_view.as_ref(),
                bias_view.as_ref(),
                group_size as i32,
                ws_ptr,
                ws_bytes,
                raw_stream,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_quant::{QuantScheme, RepackedWeights};
    use candle_core::{DType, Device};
    use half::f16;

    /// Build an f16 candle tensor from f32 values on CPU.
    fn f16_tensor(
        vals: &[f32],
        shape: &[usize],
    ) -> Tensor {
        let v: Vec<f16> = vals.iter().map(|&x| f16::from_f32(x)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    #[test]
    fn quant_gemm_ref_matches_manual_dequant() {
        // M=1, K=4, N=2, group_size=2 -> n_groups=2.
        // a = [[1,1,1,1]]
        // b_quant row0 = [0x21, 0x43] -> nibbles [1,2,3,4]
        // b_quant row1 = [0x00, 0xFF] -> nibbles [0,0,15,15]
        // scales row0 = [1,1], row1 = [2,2]; mins row0 = [0,0], row1 = [1,1].
        // dequant row0 = [1,2,3,4]; row1 = [1,1,31,31].
        // out = a @ w.T = [[1*1+1*2+1*3+1*4, 1*1+1*1+1*31+1*31]] = [[10, 64]].
        let cpu = Device::Cpu;
        let a = Tensor::from_vec(vec![1f32, 1.0, 1.0, 1.0], (1, 4), &cpu).unwrap();
        let b_quant = Tensor::from_vec(vec![0x21u8, 0x43, 0x00, 0xFF], (2, 2), &cpu).unwrap();
        let b_scales = f16_tensor(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);
        let b_mins = f16_tensor(&[0.0, 0.0, 1.0, 1.0], &[2, 2]);

        let out = quant_gemm_ref(&a, &b_quant, &b_scales, Some(&b_mins), None, 2).unwrap();
        assert_eq!(out.dims(), [1, 2]);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (got[0] - 10.0).abs() < 1e-4 && (got[1] - 64.0).abs() < 1e-4,
            "got {got:?} want [10, 64]"
        );
    }

    #[test]
    fn quant_gemm_ref_without_mins() {
        // Same weights, no mins -> symmetric formula scale*(nibble-8).
        // dequant row0 = [1*(1-8), 1*(2-8), 1*(3-8), 1*(4-8)] = [-7,-6,-5,-4]
        // dequant row1 = [2*(0-8), 2*(0-8), 2*(15-8), 2*(15-8)] = [-16,-16,14,14]
        // out = [[-7-6-5-4, -16-16+14+14]] = [[-22, -4]].
        let cpu = Device::Cpu;
        let a = Tensor::from_vec(vec![1f32, 1.0, 1.0, 1.0], (1, 4), &cpu).unwrap();
        let b_quant = Tensor::from_vec(vec![0x21u8, 0x43, 0x00, 0xFF], (2, 2), &cpu).unwrap();
        let b_scales = f16_tensor(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);

        let out = quant_gemm_ref(&a, &b_quant, &b_scales, None, None, 2).unwrap();
        assert_eq!(out.dims(), [1, 2]);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (got[0] - (-22.0)).abs() < 1e-4 && (got[1] - (-4.0)).abs() < 1e-4,
            "got {got:?} want [-22, -4]"
        );
    }

    #[test]
    fn quant_linear_forward() {
        // RepackedWeights: out=2, in=4, group_size=2.
        let rw = RepackedWeights {
            qs: vec![0x21u8, 0x43, 0x00, 0xFF],
            scales: vec![
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(2.0),
            ],
            mins: vec![
                f16::from_f32(0.0),
                f16::from_f32(0.0),
                f16::from_f32(1.0),
                f16::from_f32(1.0),
            ],
            out_features: 2,
            in_features: 4,
            group_size: 2,
            scheme: QuantScheme::Q4K,
        };
        let layer = QuantLinear::from_repacked(&rw, None, &Device::Cpu).unwrap();
        // input [1, 2, 4] = [[1,1,1,1],[1,1,1,1]] -> output [1, 2, 2].
        let x = Tensor::from_vec(vec![1f32; 8], (1, 2, 4), &Device::Cpu).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.dims(), [1, 2, 2]);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Each row -> [10, 64].
        assert!(
            (got[0] - 10.0).abs() < 1e-4 && (got[1] - 64.0).abs() < 1e-4,
            "got {got:?} want [10, 64, 10, 64]"
        );
    }
}
