//! Tensor metadata.
//!
//! [`TensorMeta`] is the rich, Rust-internal description of a tensor (no data
//! ownership). [`TensorView`] is the `#[repr(C)]` POD that actually crosses the
//! FFI boundary — it is the single source of truth for `ayaka_tensor_view_t` in
//! `include/ayaka/tensor_view.h`. The C struct must match field order and sizes
//! exactly; a `static_assert(sizeof(ayaka_tensor_view_t) == 160)` guards it.

use core::ffi::c_void;

use crate::device::{Device, DeviceKind};
use crate::dtype::DType;
use crate::flags::TensorFlags;
use crate::layout::Strides;
use crate::shape::{MAX_RANK, Shape};

/// Non-owning, fully-described tensor metadata used inside the Rust engine.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub device: Device,
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    pub flags: TensorFlags,
}

impl TensorMeta {
    /// Build a contiguous (row-major) tensor meta.
    pub fn contiguous(
        device: Device,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        let strides = Strides::contiguous(&shape);
        Self {
            device,
            dtype,
            shape,
            strides,
            flags: TensorFlags::CONTIGUOUS,
        }
    }

    #[inline]
    pub fn numel(&self) -> i64 {
        self.shape.numel()
    }

    #[inline]
    pub fn byte_size(&self) -> i64 {
        self.dtype.byte_size(self.numel())
    }

    /// Construct the FFI view over `data`. The caller guarantees `data` points
    /// at memory matching this metadata for the duration of the kernel call.
    pub fn view(
        &self,
        data: *mut c_void,
    ) -> TensorView {
        let mut shape = [0i64; MAX_RANK];
        let mut stride = [0i64; MAX_RANK];
        shape[..self.shape.rank()].copy_from_slice(self.shape.dims());
        stride[..self.shape.rank()].copy_from_slice(self.strides.as_slice());
        TensorView {
            data,
            shape,
            stride,
            rank: self.shape.rank() as i32,
            dtype: self.dtype.raw() as i32,
            device_kind: self.device.kind as i32,
            device_ordinal: self.device.ordinal as i32,
            flags: self.flags.bits(),
            reserved: 0,
        }
    }
}

/// `#[repr(C)]` POD mirror of `ayaka_tensor_view_t`.
///
/// Layout is chosen to be padding-free: an 8-byte pointer, two 8-byte-aligned
/// i64 arrays, then a block of i32 fields, then a reserved word to keep the
/// reserved word to keep the total an 8-byte multiple (160 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TensorView {
    /// Raw device (or host) pointer to element 0.
    pub data: *mut c_void,
    /// Dimension extents; only `rank` entries are valid.
    pub shape: [i64; MAX_RANK],
    /// Element strides; only `rank` entries are valid.
    pub stride: [i64; MAX_RANK],
    pub rank: i32,
    /// [`DType`] discriminant, widened to i32 for ABI stability.
    pub dtype: i32,
    /// [`DeviceKind`] discriminant.
    pub device_kind: i32,
    pub device_ordinal: i32,
    /// [`TensorFlags`] bits.
    pub flags: u32,
    /// Reserved for future ABI growth; must be 0.
    pub reserved: u32,
}

impl TensorView {
    /// A null/empty view (e.g. for optional tensors like KV scales).
    pub const NULL: Self = Self {
        data: core::ptr::null_mut(),
        shape: [0; MAX_RANK],
        stride: [0; MAX_RANK],
        rank: 0,
        dtype: 0,
        device_kind: DeviceKind::Cpu as i32,
        device_ordinal: 0,
        flags: 0,
        reserved: 0,
    };

    #[inline]
    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    #[inline]
    pub fn dtype(&self) -> Option<DType> {
        DType::from_raw(self.dtype as u8)
    }
}

// Compile-time guarantee that the layout matches the C header's static_assert.
// 8 (ptr) + 64 (shape) + 64 (stride) + 6*4 (i32/u32 fields) = 160 bytes.
const _: () = {
    assert!(core::mem::size_of::<TensorView>() == 160);
};
