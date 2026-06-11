use ayaka_core::device::Device;

pub type Result<T> = core::result::Result<T, MemoryError>;

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("invalid memory argument: {what}")]
    InvalidArgument { what: String },

    #[error("memory arithmetic overflow: {what}")]
    ArithmeticOverflow { what: String },

    #[error("device is not supported by ayaka-memory: {device:?}")]
    UnsupportedDevice { device: Device },

    #[error("CUDA support is required for this operation")]
    CudaRequired,

    #[error("host allocation failed for {bytes} bytes")]
    HostAlloc { bytes: usize },

    #[error("arena exhausted: needed {needed} bytes, remaining {remaining} bytes")]
    ArenaExhausted { needed: usize, remaining: usize },

    #[error("slab exhausted: all {num_slots} slots are allocated")]
    SlabExhausted { num_slots: usize },

    #[error("slab slot {slot} is out of range for {num_slots} slots")]
    InvalidSlot { slot: u32, num_slots: usize },

    #[error("slab slot {slot} is already free")]
    SlotAlreadyFree { slot: u32 },

    #[error("pool exhausted for class {class_bytes} bytes while requesting {requested} bytes; remaining region bytes {remaining}")]
    PoolExhausted {
        class_bytes: usize,
        requested: usize,
        remaining: usize,
    },

    #[error("workspace section name already exists: {name}")]
    WorkspaceDuplicateSection { name: String },

    #[error("workspace section id {id} is unknown")]
    WorkspaceUnknownSection { id: u32 },

    #[cfg(feature = "cuda")]
    #[error("{0}")]
    Kernel(#[from] ayaka_kernel_api::KernelError),
}

impl MemoryError {
    pub fn invalid(what: impl Into<String>) -> Self {
        Self::InvalidArgument { what: what.into() }
    }

    pub fn overflow(what: impl Into<String>) -> Self {
        Self::ArithmeticOverflow { what: what.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_argument_message_contains_context() {
        let err = MemoryError::InvalidArgument {
            what: "slot_bytes must be non-zero".to_string(),
        };
        assert!(err.to_string().contains("slot_bytes"));
    }

    #[test]
    fn unsupported_device_names_device() {
        let err = MemoryError::UnsupportedDevice {
            device: Device::CPU,
        };
        assert!(err.to_string().contains("Cpu"));
    }
}
