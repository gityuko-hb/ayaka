pub mod arena;
pub mod error;
pub mod host_buffer;
pub mod slab;
pub mod span;
pub mod stats;
pub mod stream;

pub use arena::DeviceArena;
pub use error::{MemoryError, Result};
pub use host_buffer::HostBuffer;
pub use slab::{SlabAllocator, SlabSlot};
pub use span::DeviceSpan;
pub use stats::{
    DriverMemoryInfo, KvBudget, MemoryLedger, MemoryPurpose, MemorySnapshot, MemoryStats,
    global_ledger,
};
pub use stream::StreamHandle;

#[cfg(feature = "cuda")]
pub mod copy;
#[cfg(feature = "cuda")]
pub mod device_buffer;
#[cfg(feature = "cuda")]
pub mod pinned_buffer;
#[cfg(feature = "cuda")]
pub use device_buffer::{DeviceAllocSource, DeviceBuffer};
#[cfg(feature = "cuda")]
pub use pinned_buffer::{PinnedBuffer, PinnedRing, PinnedSlot};
