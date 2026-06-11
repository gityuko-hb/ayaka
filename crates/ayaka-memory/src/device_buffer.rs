#![cfg(feature = "cuda")]

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ayaka_core::device::Device;
use ayaka_kernel_api::mem;

use crate::span::DeviceSpan;
use crate::stats::{MemoryPurpose, global_ledger};
use crate::stream::StreamHandle;
use crate::{MemoryError, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeviceAllocSource {
    Sync,
    Async { stream: StreamHandle },
}

pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
    device: Device,
    source: DeviceAllocSource,
    purpose: MemoryPurpose,
}

impl DeviceBuffer {
    pub fn reserve(
        device: Device,
        bytes: usize,
    ) -> Result<Self> {
        Self::reserve_for(device, bytes, MemoryPurpose::Transient)
    }

    pub fn reserve_for(
        device: Device,
        bytes: usize,
        purpose: MemoryPurpose,
    ) -> Result<Self> {
        let ordinal = cuda_ordinal(device)?;
        let ptr = if bytes == 0 {
            0
        } else {
            unsafe { mem::device_alloc(bytes, ordinal)? as usize as u64 }
        };
        global_ledger().register(purpose, bytes);
        Ok(Self {
            ptr,
            len: bytes,
            device,
            source: DeviceAllocSource::Sync,
            purpose,
        })
    }

    pub fn alloc_async(
        device: Device,
        bytes: usize,
        stream: StreamHandle,
    ) -> Result<Self> {
        Self::alloc_async_for(device, bytes, stream, MemoryPurpose::Transient)
    }

    pub fn alloc_async_for(
        device: Device,
        bytes: usize,
        stream: StreamHandle,
        purpose: MemoryPurpose,
    ) -> Result<Self> {
        let supported = mem_pool_supported_cached(device)?;
        Self::alloc_async_with_pool_support(device, bytes, stream, purpose, supported)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn source(&self) -> DeviceAllocSource {
        self.source
    }

    pub fn span(&self) -> DeviceSpan {
        DeviceSpan::new(self.ptr, self.len)
    }

    pub fn free_on(
        mut self,
        stream: StreamHandle,
    ) -> Result<()> {
        self.release(Some(stream))
    }

    #[cfg(test)]
    pub(crate) fn alloc_async_with_pool_support_for_test(
        device: Device,
        bytes: usize,
        stream: StreamHandle,
        purpose: MemoryPurpose,
        supported: bool,
    ) -> Result<Self> {
        Self::alloc_async_with_pool_support(device, bytes, stream, purpose, supported)
    }

    fn alloc_async_with_pool_support(
        device: Device,
        bytes: usize,
        stream: StreamHandle,
        purpose: MemoryPurpose,
        supported: bool,
    ) -> Result<Self> {
        if !supported {
            return Self::reserve_for(device, bytes, purpose);
        }

        let ordinal = cuda_ordinal(device)?;
        let ptr = if bytes == 0 {
            0
        } else {
            unsafe { mem::device_alloc_async(bytes, ordinal, stream.as_ffi())? as usize as u64 }
        };
        global_ledger().register(purpose, bytes);
        Ok(Self {
            ptr,
            len: bytes,
            device,
            source: DeviceAllocSource::Async { stream },
            purpose,
        })
    }

    fn release(
        &mut self,
        free_stream: Option<StreamHandle>,
    ) -> Result<()> {
        if self.ptr == 0 {
            return Ok(());
        }
        let ordinal = cuda_ordinal(self.device)?;
        let ptr = self.ptr as usize as *mut c_void;
        match self.source {
            DeviceAllocSource::Sync => unsafe {
                mem::device_free(ptr, ordinal)?;
            },
            DeviceAllocSource::Async { stream } => {
                let stream = free_stream.unwrap_or(stream);
                unsafe {
                    mem::device_free_async(ptr, ordinal, stream.as_ffi())?;
                }
            },
        }
        self.ptr = 0;
        let len = self.len;
        self.len = 0;
        global_ledger().deregister(self.purpose, len)?;
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Err(err) = self.release(None) {
            tracing::error!(error = %err, "leaking device buffer after failed free");
        }
    }
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

fn cuda_ordinal(device: Device) -> Result<i32> {
    if !device.is_cuda() {
        return Err(MemoryError::UnsupportedDevice { device });
    }
    Ok(i32::from(device.ordinal))
}

fn mem_pool_supported_cached(device: Device) -> Result<bool> {
    let ordinal = cuda_ordinal(device)?;
    static CACHE: OnceLock<Mutex<HashMap<u16, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache
            .lock()
            .map_err(|_| MemoryError::invalid("memory pool support cache lock poisoned"))?;
        if let Some(&supported) = guard.get(&device.ordinal) {
            return Ok(supported);
        }
    }

    let supported = mem::pool_supported(ordinal)?;
    let mut guard = cache
        .lock()
        .map_err(|_| MemoryError::invalid("memory pool support cache lock poisoned"))?;
    guard.insert(device.ordinal, supported);
    Ok(supported)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_reserve_records_span_and_source() {
        let buffer =
            DeviceBuffer::reserve_for(Device::cuda(0), 1024, MemoryPurpose::Transient).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.span().is_empty());
        assert_eq!(buffer.source(), DeviceAllocSource::Sync);
    }

    #[test]
    fn forced_async_fallback_records_sync_source() {
        let buffer = DeviceBuffer::alloc_async_with_pool_support_for_test(
            Device::cuda(0),
            1024,
            StreamHandle::null(),
            MemoryPurpose::Transient,
            false,
        )
        .unwrap();
        assert_eq!(buffer.source(), DeviceAllocSource::Sync);
    }
}
