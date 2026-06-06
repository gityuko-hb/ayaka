//! Numeric data types understood by the engine and the kernels.
//!
//! `#[repr(u8)]` with explicit discriminants — these MUST match
//! `ayaka_dtype_t` in `include/ayaka/dtype.h`.

/// Element data type. Covers the float, fp8/fp4, integer and sub-byte formats an
/// inference engine needs across Ampere → Blackwell.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum DType {
    /// IEEE-754 single precision.
    F32 = 0,

    /// IEEE-754 half precision.
    F16 = 1,

    /// Brain Float 16 (NV's alternative to FP16 with wider exponent, same 8-bit mantissa). Ampere+.
    BF16 = 2,

    /// FP8 E4M3 (activations / weights). Max magnitude 448.
    F8E4M3 = 3,

    /// FP8 E5M2 (wider range, lower mantissa).
    F8E5M2 = 4,

    /// FP4 E2M1 element (NVFP4 / MXFP4 payload). Blackwell.
    F4E2M1 = 5,

    /// FP8 E8M0 scale type for micro-scaled (MX) formats.
    F8E8M0 = 6,

    /// Signed 64-bit integer.
    I64 = 7,

    /// Signed 32-bit integer.
    I32 = 8,

    /// Signed 16-bit integer.
    I16 = 9,

    /// Signed 8-bit integer.
    I8 = 10,

    /// Unsigned 8-bit integer.
    U8 = 11,

    /// Signed 4-bit, two elements packed per byte.
    I4 = 12,

    /// Unsigned 4-bit, two elements packed per byte.
    U4 = 13,

    /// Boolean, stored one-per-byte.
    Bool = 14,
}

impl DType {
    /// Number of bits a single element occupies (sub-byte types return 4).
    pub const fn bits(self) -> u32 {
        match self {
            DType::F32 | DType::I32 => 32,
            DType::I64 => 64,
            DType::F16 | DType::BF16 | DType::I16 => 16,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 | DType::I8 | DType::U8 | DType::Bool => {
                8
            },
            DType::F4E2M1 | DType::I4 | DType::U4 => 4,
        }
    }

    /// Byte size of `numel` elements, accounting for sub-byte packing.
    pub const fn byte_size(
        self,
        numel: i64,
    ) -> i64 {
        (numel * self.bits() as i64 + 7) / 8
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2 | DType::F4E2M1
        )
    }

    pub const fn is_float8(self) -> bool {
        matches!(self, DType::F8E4M3 | DType::F8E5M2)
    }

    pub const fn is_float4(self) -> bool {
        matches!(self, DType::F4E2M1)
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U8 | DType::I4 | DType::U4
        )
    }

    pub const fn is_sub_byte(self) -> bool {
        matches!(self, DType::F4E2M1 | DType::I4 | DType::U4)
    }

    /// True for any reduced-precision storage format used for weights / KV.
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            DType::F8E4M3
                | DType::F8E5M2
                | DType::F4E2M1
                | DType::I8
                | DType::U8
                | DType::I4
                | DType::U4
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8E4M3 => "f8e4m3",
            DType::F8E5M2 => "f8e5m2",
            DType::F4E2M1 => "f4e2m1",
            DType::F8E8M0 => "f8e8m0",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::I4 => "i4",
            DType::U4 => "u4",
            DType::Bool => "bool",
        }
    }

    pub const fn from_raw(v: u8) -> Option<Self> {
        Some(match v {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::BF16,
            3 => DType::F8E4M3,
            4 => DType::F8E5M2,
            5 => DType::F4E2M1,
            6 => DType::F8E8M0,
            7 => DType::I64,
            8 => DType::I32,
            9 => DType::I16,
            10 => DType::I8,
            11 => DType::U8,
            12 => DType::I4,
            13 => DType::U4,
            14 => DType::Bool,
            _ => return None,
        })
    }

    #[inline]
    pub const fn raw(self) -> u8 {
        self as u8
    }
}
