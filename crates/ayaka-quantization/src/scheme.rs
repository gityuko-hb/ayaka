//! On-disk weight quantization schemes and their per-weight byte cost.

/// How a tensor's weights are stored on disk.
///
/// `bytes_per_weight` is the amortized storage cost of a single scalar weight,
/// including any per-block scales/mins. These figures drive memory estimation
/// in `ayaka-loader::estimate`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// IEEE half precision (2 bytes/weight).
    F16,
    /// bfloat16 (2 bytes/weight).
    BF16,
    /// GGUF Q8_0: blocks of 32, one f16 scale per block (34 bytes/block).
    Q8_0,
    /// GGUF Q4_0: blocks of 32, one f16 scale per block (18 bytes/block).
    Q4_0,
    /// GGUF Q4_K: super-blocks of 256 (144 bytes/super-block).
    Q4K,
    /// MXFP4: blocks of 32 with a shared E8M0 byte scale (17 bytes/block).
    MXFP4,
}

impl QuantScheme {
    /// Amortized bytes used to store a single weight, including block scales.
    ///
    /// Matches the design doc §1.1 table:
    /// Q4_0 = 0.5625, Q8_0 = 1.0625, Q4_K = 0.5625, MXFP4 = 0.53125.
    pub const fn bytes_per_weight(self) -> f64 {
        match self {
            QuantScheme::F16 | QuantScheme::BF16 => 2.0,
            // 32 i8 + 1 f16 scale = 34 bytes / 32 weights.
            QuantScheme::Q8_0 => 34.0 / 32.0,
            // 16 packed nibbles + 1 f16 scale = 18 bytes / 32 weights.
            QuantScheme::Q4_0 => 18.0 / 32.0,
            // 144-byte super-block / 256 weights = 4.5 bits/weight.
            QuantScheme::Q4K => 144.0 / 256.0,
            // 16 packed nibbles + 1 E8M0 byte scale = 17 bytes / 32 weights.
            QuantScheme::MXFP4 => 17.0 / 32.0,
        }
    }

    /// `true` when the scheme is a plain (non-quantized) floating layout that
    /// kernels can consume directly without dequantization.
    pub const fn is_float(self) -> bool {
        matches!(self, QuantScheme::F16 | QuantScheme::BF16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_weight_matches_design_doc() {
        assert_eq!(QuantScheme::F16.bytes_per_weight(), 2.0);
        assert_eq!(QuantScheme::BF16.bytes_per_weight(), 2.0);
        assert_eq!(QuantScheme::Q8_0.bytes_per_weight(), 1.0625);
        assert_eq!(QuantScheme::Q4_0.bytes_per_weight(), 0.5625);
        assert_eq!(QuantScheme::Q4K.bytes_per_weight(), 0.5625);
        assert_eq!(QuantScheme::MXFP4.bytes_per_weight(), 0.53125);
    }

    #[test]
    fn only_f16_bf16_are_float() {
        assert!(QuantScheme::F16.is_float());
        assert!(QuantScheme::BF16.is_float());
        assert!(!QuantScheme::Q8_0.is_float());
        assert!(!QuantScheme::Q4_0.is_float());
        assert!(!QuantScheme::MXFP4.is_float());
    }
}
