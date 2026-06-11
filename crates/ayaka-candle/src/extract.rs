//! Conversion from candle tensors to ayaka FFI descriptors.
//!
//! The only place where candle storage internals and cudarc streams are
//! visible. Everything below this layer speaks `TensorView`.

use std::sync::Arc;

use ayaka_core::device::Device as AyakaDevice;
use ayaka_core::dtype::DType as AyakaDType;
use ayaka_core::shape::Shape as AyakaShape;
use ayaka_core::tensor_meta::{TensorMeta, TensorView};
use candle_core::cuda_backend::CudaStorage;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::{DType, Device, DeviceLocation, Layout, Result, Storage};

/// Map a candle dtype onto the ayaka ABI dtype.
pub fn ayaka_dtype(dtype: DType) -> Result<AyakaDType> {
    match dtype {
        DType::F32 => Ok(AyakaDType::F32),
        DType::F16 => Ok(AyakaDType::F16),
        DType::BF16 => Ok(AyakaDType::BF16),
        dt => candle_core::bail!("ayaka kernels do not support dtype {dt:?}"),
    }
}

/// The stream candle issues its own work on for `device`; launching on the
/// same stream orders ayaka kernels after candle's pending ops.
pub fn cuda_stream(device: &Device) -> Result<Arc<CudaStream>> {
    match device {
        Device::Cuda(dev) => Ok(dev.cuda_stream()),
        _ => candle_core::bail!("ayaka kernels require a CUDA device"),
    }
}

pub fn cuda_ordinal(device: &Device) -> Result<u16> {
    match device.location() {
        DeviceLocation::Cuda { gpu_id } => Ok(gpu_id as u16),
        _ => candle_core::bail!("ayaka kernels require a CUDA device"),
    }
}

/// Borrow the CUDA storage out of a locked candle storage.
pub fn cuda_storage<'a>(
    storage: &'a Storage,
    what: &str,
) -> Result<&'a CudaStorage> {
    match storage {
        Storage::Cuda(s) => Ok(s),
        _ => candle_core::bail!("{what}: ayaka kernels require CUDA tensors"),
    }
}

/// Require row-major contiguity; the kernels only take the fast path for now.
pub fn require_contiguous(
    layout: &Layout,
    what: &str,
) -> Result<()> {
    if !layout.is_contiguous() {
        candle_core::bail!("{what}: ayaka kernels require contiguous tensors");
    }
    Ok(())
}

/// Build the FFI view for a contiguous tensor whose first element lives at
/// device pointer `ptr` (already offset by the layout's start offset).
pub fn contiguous_view(
    layout: &Layout,
    dtype: AyakaDType,
    ordinal: u16,
    ptr: u64,
) -> TensorView {
    let dims: Vec<i64> = layout.dims().iter().map(|&d| d as i64).collect();
    TensorMeta::contiguous(AyakaDevice::cuda(ordinal), dtype, AyakaShape::new(&dims))
        .view(ptr as usize as *mut core::ffi::c_void)
}
