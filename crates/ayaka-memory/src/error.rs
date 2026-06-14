//! Error helpers for `ayaka-memory`.
//!
//! All public fallible operations in this crate return `ayaka_error::Result<T>`,
//! using `AyakaError` with an `ErrorKind::Memory` or `ErrorKind::Cuda` domain.

use ayaka_core::device::Device;
use ayaka_error::{AyakaError, ErrorKind};

/// Returns an `InvalidArgument` error with context.
pub fn invalid(what: impl Into<String>) -> AyakaError {
    AyakaError::invalid_argument(format!("invalid memory argument: {}", what.into()))
}

/// Returns a `Memory` error for arithmetic overflow.
pub fn overflow(what: impl Into<String>) -> AyakaError {
    AyakaError::new(
        ErrorKind::Memory,
        format!("memory arithmetic overflow: {}", what.into()),
    )
}

/// Returns an error for an unsupported device.
pub fn unsupported_device(device: Device) -> AyakaError {
    AyakaError::new(
        ErrorKind::Device,
        format!("device is not supported by ayaka-memory: {device:?}"),
    )
}

/// Returns an error when CUDA support is required but unavailable.
pub fn cuda_required() -> AyakaError {
    AyakaError::new(
        ErrorKind::Cuda,
        "CUDA support is required for this operation",
    )
}

/// Returns an error for a failed host allocation.
pub fn host_alloc(bytes: usize) -> AyakaError {
    AyakaError::out_of_memory(format!("host allocation failed for {bytes} bytes"))
}

/// Returns an error for an exhausted arena.
pub fn arena_exhausted(
    needed: usize,
    remaining: usize,
) -> AyakaError {
    AyakaError::out_of_memory(format!(
        "arena exhausted: needed {needed} bytes, remaining {remaining} bytes"
    ))
}

/// Returns an error for an exhausted slab.
pub fn slab_exhausted(num_slots: usize) -> AyakaError {
    AyakaError::out_of_memory(format!(
        "slab exhausted: all {num_slots} slots are allocated"
    ))
}

/// Returns an error for an invalid slab slot.
pub fn invalid_slot(
    slot: u32,
    num_slots: usize,
) -> AyakaError {
    AyakaError::invalid_argument(format!(
        "slab slot {slot} is out of range for {num_slots} slots"
    ))
}

/// Returns an error for freeing an already-free slab slot.
pub fn slot_already_free(slot: u32) -> AyakaError {
    AyakaError::invalid_argument(format!("slab slot {slot} is already free"))
}

/// Returns an error for an exhausted size-class pool.
pub fn pool_exhausted(
    class_bytes: usize,
    requested: usize,
    remaining: usize,
) -> AyakaError {
    AyakaError::out_of_memory(format!(
        "pool exhausted for class {class_bytes} bytes while requesting {requested} bytes; remaining region bytes {remaining}"
    ))
}

/// Returns an error for a duplicate workspace section name.
pub fn workspace_duplicate_section(name: impl Into<String>) -> AyakaError {
    AyakaError::invalid_argument(format!(
        "workspace section name already exists: {}",
        name.into()
    ))
}

/// Returns an error for an unknown workspace section id.
pub fn workspace_unknown_section(id: u32) -> AyakaError {
    AyakaError::invalid_argument(format!("workspace section id {id} is unknown"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_argument_message_contains_context() {
        let err = invalid("slot_bytes must be non-zero");
        assert!(err.to_string().contains("slot_bytes"));
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn unsupported_device_names_device() {
        let err = unsupported_device(Device::CPU);
        assert!(err.to_string().contains("Cpu"));
        assert_eq!(err.kind(), ErrorKind::Device);
    }
}
