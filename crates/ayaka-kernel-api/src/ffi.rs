use core::ffi::c_char;

pub use ayaka_core::device::{
    Device as AyakaDevice, DeviceKind as AyakaDeviceKind, SmArch as AyakaSmArch,
};
pub use ayaka_core::dtype::DType as AyakaDType;
pub use ayaka_core::shape::MAX_RANK as AYAKA_MAX_RANK;
pub use ayaka_core::tensor_meta::{ABI_VERSION, TensorView as AyakaTensorView};

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AyakaStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    DTypeError = 2,
    ShapeError = 3,
    DeviceError = 4,
    Unsupported = 5,
    OutOfMemory = 6,
    ContextLengthExceeded = 7,
    NotFound = 8,
    AbiMismatch = 9,
    KernelLaunchError = 10,
    StreamError = 11,
    InternalError = 12,
    Cancelled = 13,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct AyakaStatus {
    pub code: AyakaStatusCode,
    pub message: *const c_char,
}

pub type AyakaStream = *mut core::ffi::c_void;

/// Rotary pair layout. Mirrors `ayaka_rope_layout_t` in
/// `native/include/ayaka/ops/rope.h`.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RopeLayout {
    /// Rotate-halves (GPT-NeoX / Llama): pairs `(d, d + rot_dim/2)`.
    Neox = 0,
    /// Interleaved (GPT-J): pairs `(2d, 2d + 1)`.
    GptJ = 1,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// `ayaka_rmsnorm` in `native/include/ayaka/ops/rmsnorm.h`.
    pub fn ayaka_rmsnorm(
        out: *const AyakaTensorView,
        input: *const AyakaTensorView,
        weight: *const AyakaTensorView,
        eps: f32,
        stream: AyakaStream,
    ) -> AyakaStatus;

    /// `ayaka_fused_add_rmsnorm` in `native/include/ayaka/ops/fused_add_rmsnorm.h`.
    pub fn ayaka_fused_add_rmsnorm(
        out: *const AyakaTensorView,
        residual_out: *const AyakaTensorView,
        input: *const AyakaTensorView,
        residual: *const AyakaTensorView,
        weight: *const AyakaTensorView,
        eps: f32,
        stream: AyakaStream,
    ) -> AyakaStatus;

    /// `ayaka_rope` in `native/include/ayaka/ops/rope.h`.
    pub fn ayaka_rope(
        query: *const AyakaTensorView,
        key: *const AyakaTensorView,
        positions: *const AyakaTensorView,
        cos_sin_cache: *const AyakaTensorView,
        layout: i32,
        stream: AyakaStream,
    ) -> AyakaStatus;

    /// `ayaka_silu_and_mul` in `native/include/ayaka/ops/silu_and_mul.h`.
    pub fn ayaka_silu_and_mul(
        out: *const AyakaTensorView,
        input: *const AyakaTensorView,
        stream: AyakaStream,
    ) -> AyakaStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_core::tensor_meta::TensorView;

    #[test]
    fn test_tensor_view_size_matches_c() {
        assert_eq!(core::mem::size_of::<TensorView>(), 176);
    }

    #[test]
    fn test_tensor_view_align() {
        assert!(core::mem::align_of::<TensorView>() <= 8);
    }

    #[test]
    fn test_dtype_discriminants_match_header() {
        assert_eq!(AyakaDType::F32 as i32, 0);
        assert_eq!(AyakaDType::F16 as i32, 1);
        assert_eq!(AyakaDType::BF16 as i32, 2);
        assert_eq!(AyakaDType::Bool as i32, 14);
    }

    #[test]
    fn test_device_kind_discriminants() {
        assert_eq!(AyakaDeviceKind::Cpu as i32, 0);
        assert_eq!(AyakaDeviceKind::Cuda as i32, 1);
    }

    #[test]
    fn test_status_code_discriminants() {
        assert_eq!(AyakaStatusCode::Ok as i32, 0);
        assert_eq!(AyakaStatusCode::Cancelled as i32, 13);
    }

    #[test]
    fn test_rope_layout_discriminants_match_header() {
        assert_eq!(RopeLayout::Neox as i32, 0);
        assert_eq!(RopeLayout::GptJ as i32, 1);
    }
}
