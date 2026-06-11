//! Tensor metadata.
//!
//! [`TensorMeta`] is the rich, Rust-internal description of a tensor (no data
//! ownership). [`TensorView`] is the `#[repr(C)]` POD that actually crosses the
//! FFI boundary — it is the single source of truth for `ayaka_tensor_view_t` in
//! `include/ayaka/tensor_view.h`. The C struct must match field order and sizes
//! exactly; a `static_assert(sizeof(ayaka_tensor_view_t) == 176)` guards it.
//!
//! Every FFI-crossing descriptor opens with `AYAKA_DESC_HEADER` (struct_size +
//! abi_version) for sized-struct versioning. Future fields go into `_reserved[]`
//! at the end with a MINOR version bump.

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
            struct_size: core::mem::size_of::<TensorView>() as u32,
            abi_version: ABI_VERSION,
            data,
            shape,
            stride,
            rank: self.shape.rank() as i32,
            dtype: self.dtype.raw() as i32,
            device_kind: self.device.kind as i32,
            device_ordinal: self.device.ordinal as i32,
            flags: self.flags.bits(),
            _reserved: [0; 3],
        }
    }
}

/// `#[repr(C)]` POD mirror of `ayaka_tensor_view_t`.
///
/// Opens with [`AYAKA_DESC_HEADER`] for sized-struct ABI versioning.
/// Layout: 4(struct_size) + 4(abi_version) + 8(data) + 64(shape) + 64(stride)
/// + 4(rank) + 4(dtype) + 4(device_kind) + 4(device_ordinal) + 4(flags)
/// + 12(_reserved[3]) = 176 bytes, aligned to 8.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TensorView {
    pub struct_size: u32,
    pub abi_version: u32,
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
    /// Reserved for future ABI growth via MINOR version bumps.
    pub _reserved: [u32; 3],
}

pub const ABI_VERSION: u32 = 1;

impl TensorView {
    /// A null/empty view (e.g. for optional tensors like KV scales).
    pub const NULL: Self = Self {
        struct_size: core::mem::size_of::<Self>() as u32,
        abi_version: ABI_VERSION,
        data: core::ptr::null_mut(),
        shape: [0; MAX_RANK],
        stride: [0; MAX_RANK],
        rank: 0,
        dtype: 0,
        device_kind: DeviceKind::Cpu as i32,
        device_ordinal: 0,
        flags: 0,
        _reserved: [0; 3],
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
// 4+4 + 8 + 64 + 64 + 4+4+4+4+4 + 12 = 176 bytes.
const _: () = {
    assert!(core::mem::size_of::<TensorView>() == 176);
};
