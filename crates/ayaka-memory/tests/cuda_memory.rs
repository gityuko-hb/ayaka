#![cfg(feature = "cuda")]

use ayaka_core::device::Device;
use ayaka_kernel_api::mem;
use ayaka_memory::copy;
use ayaka_memory::{
    DeviceBuffer, KvBudget, MemoryPurpose, PinnedRing, StreamHandle, global_ledger,
};

#[test]
fn device_copy_roundtrip_default_stream() {
    let device = Device::cuda(0);
    let stream = StreamHandle::null();
    let input = b"ayaka-memory-roundtrip";
    let buffer = DeviceBuffer::reserve_for(device, input.len(), MemoryPurpose::Transient).unwrap();

    copy::h2d_async(buffer.span(), input, stream).unwrap();
    let mut output = vec![0u8; input.len()];
    copy::d2h_async(&mut output, buffer.span(), stream).unwrap();
    mem::stream_synchronize(stream.as_ffi()).unwrap();

    assert_eq!(output, input);
}

#[test]
fn pinned_ring_h2d_roundtrip() {
    let device = Device::cuda(0);
    let stream = StreamHandle::null();
    let input = b"pinned-ring-copy";
    let mut ring = PinnedRing::new(2, 64).unwrap();
    let buffer = DeviceBuffer::reserve_for(device, input.len(), MemoryPurpose::Transient).unwrap();

    let written = {
        let mut slot = ring.acquire(0);
        slot.bump_write(input, 1).unwrap().to_vec()
    };
    copy::h2d_async(buffer.span(), &written, stream).unwrap();
    let mut output = vec![0u8; input.len()];
    copy::d2h_async(&mut output, buffer.span(), stream).unwrap();
    mem::stream_synchronize(stream.as_ffi()).unwrap();

    assert_eq!(output, input);
}

#[test]
fn ledger_returns_to_baseline_after_device_buffer_drop() {
    let device = Device::cuda(0);
    let ledger = global_ledger();
    let before = ledger.snapshot().transient;
    {
        let buffer = DeviceBuffer::reserve_for(device, 4096, MemoryPurpose::Transient).unwrap();
        assert_eq!(buffer.len(), 4096);
        assert!(ledger.snapshot().transient >= before + 4096);
    }
    assert_eq!(ledger.snapshot().transient, before);
}

#[test]
fn driver_info_and_pool_support_smoke() {
    let info = mem::get_info(0).unwrap();
    assert!(info.total_bytes > 0);
    assert!(info.free_bytes <= info.total_bytes);

    let _supported = mem::pool_supported(0).unwrap();
    mem::pool_trim(0, 0).unwrap();
}

#[test]
fn budget_sequence_arithmetic_smoke() {
    let info = mem::get_info(0).unwrap();
    let block_bytes = 16 * 1024;
    let budget = KvBudget::from_free_memory(info.free_bytes, info.total_bytes, block_bytes, 0.05)
        .unwrap();
    assert_eq!(budget.block_bytes, block_bytes);
    assert_eq!(budget.num_blocks, budget.kv_budget_bytes / block_bytes);
}
