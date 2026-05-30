#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum Dtype {
    // Boolean.
    Bool = 0,

    // Unsigned 8 bits integer.
    U8 = 1,
    // Unsigned 16 bits integer.
    U16 = 2,
    // Unsigned 32 bits integer.
    U32 = 3,
    // Unsigned 64 bits integer.
    U64 = 4,

    // Signed 8 bits integer.
    I8 = 10,
    // Signed 16 bits integer.
    I16 = 11,
    // Signed 32 bits integer.
    I32 = 12,
    // Signed 64 bits integer.
    I64 = 13,

    // Floating-point using 16 bits = half-precision = 1 sign bit + 5 exponent bits + 10 mantissa bits.
    F16 = 20,
    // Brain Floating-point using 16 bits = bfloat16 = 1 sign bit + 8 exponent bits + 7 mantissa bits.
    BF16 = 21,
    // Floating-point using 32 bits = single-precision = 1 sign bit + 8 exponent bits + 23 mantissa bits.
    F32 = 22,
    // Floating-point using 64 bits = double-precision = 1 sign bit + 11 exponent bits + 52 mantissa bits.
    F64 = 23,

    // Floating-point 8 bit with 4 exponent bits and 3 mantissa bits.
    FP8E4M3 = 30,
    // Floating-point 8 bit with 5 exponent bits and 2 mantissa bits.
    FP8E5M2 = 31,
    // Floating-point 6 bit with 2 exponent bits and 3 mantissa bits.
    FP6E2M3 = 32,
    // Floating-point 6 bit with 3 exponent bits and 2 mantissa bits.
    FP6E3M2 = 33,
    // Floating-point 8 bit with 5 exponent bits and 3 mantissa bits.
    F8E5M3 = 34,
    // Floating-point 8 bit with 8 exponent bits and 0 mantissa bits.
    F8E8M0 = 35,

    // 8-bit quantized integer, used for reducing memory footprint and accelerating inference with minimal precision loss.
    INT8 = 40,
    // 4-bit quantized integer, used for extreme memory compression; typically requires group quantization (e.g., AWQ, GPTQ).
    INT4 = 41,
    // 4-bit NormalFloat, an information-theoretically optimal data type for normally distributed weights, famously used in QLoRA.
    NF4 = 42,
    // 4-bit Microscaling format (OCP MX4), utilizing block-level shared scaling factors for highly efficient hardware matrix math.
    MX4 = 43,
}

impl Dtype {
    #[inline]
    pub const fn size_in_bytes(self) -> Option<usize> {
        let b = self.bits();
        if b.is_multiple_of(8) {
            Some(b / 8)
        } else {
            None
        }
    }

    #[inline]
    pub const fn bits(self) -> usize {
        match self {
            Self::Bool => 8,

            Self::U8
            | Self::I8
            | Self::FP8E4M3
            | Self::FP8E5M2
            | Self::F8E5M3
            | Self::F8E8M0
            | Self::INT8 => 8,

            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 16,

            Self::U32 | Self::I32 | Self::F32 => 32,

            Self::U64 | Self::I64 | Self::F64 => 64,

            Self::FP6E2M3 | Self::FP6E3M2 => 6,

            Self::INT4 | Self::NF4 | Self::MX4 => 4,
        }
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::BF16
                | Self::F32
                | Self::F64
                | Self::FP8E4M3
                | Self::FP8E5M2
                | Self::FP6E2M3
                | Self::FP6E3M2
                | Self::F8E5M3
                | Self::F8E8M0
        )
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::INT8 | Self::INT4 | Self::NF4 | Self::MX4)
    }

    /// Check if the data type is packed.
    /// Types that are not divisible by 8 (such as 4-bit, 6-bit) cannot stand alone.
    /// In a single byte, they must be "packed" together.
    pub fn is_packed(self) -> bool {
        self.size_in_bytes().is_none()
    }
}
