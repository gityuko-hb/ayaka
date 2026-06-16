//! Live CUDA driver memory query.
//!
//! `ayaka_memory::DriverMemoryInfo` is a plain struct that nothing populates;
//! this fills it. Mirrors the pattern in `ayaka-utils::memory_usage` so both
//! sites agree on how free/total VRAM is read.

use ayaka_memory::DriverMemoryInfo;
use candle_core::Device;

use crate::error::{LoaderError, Result};

/// Query free/total VRAM for `device` (must be a CUDA device).
pub fn driver_memory_info(device: &Device) -> Result<DriverMemoryInfo> {
    match device {
        Device::Cuda(dev) => {
            use candle_core::cuda::cudarc::driver::result;
            use candle_core::cuda_backend::WrapErr;

            dev.cuda_stream().context().bind_to_thread().w()?;
            let (free_bytes, total_bytes) = result::mem_get_info().w()?;
            Ok(DriverMemoryInfo {
                free_bytes,
                total_bytes,
            })
        },
        _ => Err(LoaderError::InvalidConfig(
            "driver_memory_info requires a CUDA device".to_string(),
        )),
    }
}
