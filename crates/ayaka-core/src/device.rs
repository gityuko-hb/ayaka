use crate::id::DeviceId;

#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceKind {
    Cpu = 0,
    Cuda = 1,
    Metal = 2,
    Rocm = 3,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Device {
    pub kind: DeviceKind,
    pub id: DeviceId,
}

impl Device {
    pub const CPU: Self = Self {
        kind: DeviceKind::Cpu,
        id: DeviceId::ZERO,
    };

    #[inline]
    pub const fn cuda(index: u16) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            id: DeviceId::new(index),
        }
    }

    #[inline]
    pub const fn is_cuda(self) -> bool {
        matches!(self.kind, DeviceKind::Cuda)
    }

    #[inline]
    pub const fn is_cpu(self) -> bool {
        matches!(self.kind, DeviceKind::Cpu)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeviceInfo {
    pub device: Device,
    pub name: String,
    pub total_memory_bytes: u64,
    pub compute_capability_major: u16,
    pub compute_capability_minor: u16,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
}
