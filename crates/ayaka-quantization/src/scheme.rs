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
    /// GGUF Q6_K: super-blocks of 256 (210 bytes/super-block).
    Q6K,
    /// MXFP4: blocks of 32 with a shared E8M0 byte scale (17 bytes/block).
    MXFP4,
    /// AWQ quantization (HuggingFace safetensors, int4 with per-group scales + zeros).
    /// Canonical group_size = 128 → bytes_per_weight ≈ 0.53125.
    AWQ,
    /// GPTQ quantization (HuggingFace safetensors, int4 with per-group scales + zeros).
    /// Canonical group_size = 128 → bytes_per_weight ≈ 0.53125.
    GPTQ,
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
            // 210-byte super-block / 256 weights = 6.5 bits/weight.
            QuantScheme::Q6K => 210.0 / 256.0,
            // 16 packed nibbles + 1 E8M0 byte scale = 17 bytes / 32 weights.
            QuantScheme::MXFP4 => 17.0 / 32.0,
            // AWQ/GPTQ: 4-bit packed (0.5 bytes) + f16 scale + f16 zero per 128 weights.
            // = 0.5 + 2*(2/128) = 0.53125
            QuantScheme::AWQ | QuantScheme::GPTQ => 0.53125,
        }
    }

    /// `true` when the scheme is a plain (non-quantized) floating layout that
    /// kernels can consume directly without dequantization.
    pub const fn is_float(self) -> bool {
        matches!(self, QuantScheme::F16 | QuantScheme::BF16)
    }

    /// Map this scheme to its corresponding [`GgufDtype`](crate::gguf_block::GgufDtype),
    /// if it has one.
    ///
    /// Returns `None` for non-GGUF schemes (AWQ, GPTQ), which come from
    /// HuggingFace safetensors and have no `ggml` type id.
    pub const fn try_to_gguf_dtype(self) -> Option<crate::gguf_block::GgufDtype> {
        match self {
            QuantScheme::F16 => Some(crate::gguf_block::GgufDtype::F16),
            QuantScheme::BF16 => Some(crate::gguf_block::GgufDtype::BF16),
            QuantScheme::Q8_0 => Some(crate::gguf_block::GgufDtype::Q8_0),
            QuantScheme::Q4_0 => Some(crate::gguf_block::GgufDtype::Q4_0),
            QuantScheme::Q4K => Some(crate::gguf_block::GgufDtype::Q4K),
            QuantScheme::Q6K => Some(crate::gguf_block::GgufDtype::Q6K),
            QuantScheme::MXFP4 => Some(crate::gguf_block::GgufDtype::MXFP4),
            QuantScheme::AWQ | QuantScheme::GPTQ => None,
        }
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
        assert_eq!(QuantScheme::Q6K.bytes_per_weight(), 210.0 / 256.0);
        assert_eq!(QuantScheme::MXFP4.bytes_per_weight(), 0.53125);
    }

    #[test]
    fn q6k_scheme_bytes_per_weight() {
        assert_eq!(QuantScheme::Q6K.bytes_per_weight(), 210.0 / 256.0);
    }

    #[test]
    fn only_f16_bf16_are_float() {
        assert!(QuantScheme::F16.is_float());
        assert!(QuantScheme::BF16.is_float());
        assert!(!QuantScheme::Q8_0.is_float());
        assert!(!QuantScheme::Q4_0.is_float());
        assert!(!QuantScheme::MXFP4.is_float());
    }

    #[test]
    fn try_to_gguf_dtype_round_trips_gguf_schemes() {
        for scheme in [
            QuantScheme::F16,
            QuantScheme::BF16,
            QuantScheme::Q8_0,
            QuantScheme::Q4_0,
            QuantScheme::Q4K,
            QuantScheme::Q6K,
            QuantScheme::MXFP4,
        ] {
            let dtype = scheme.try_to_gguf_dtype().unwrap();
            assert_eq!(dtype.scheme(), scheme, "round-trip failed for {scheme:?}");
        }
        // AWQ/GPTQ have no GGUF dtype.
        assert_eq!(QuantScheme::AWQ.try_to_gguf_dtype(), None);
        assert_eq!(QuantScheme::GPTQ.try_to_gguf_dtype(), None);
    }

    #[test]
    fn awq_gptq_bytes_per_weight() {
        assert_eq!(QuantScheme::AWQ.bytes_per_weight(), 0.53125);
        assert_eq!(QuantScheme::GPTQ.bytes_per_weight(), 0.53125);
    }

    #[test]
    fn awq_gptq_not_float() {
        assert!(!QuantScheme::AWQ.is_float());
        assert!(!QuantScheme::GPTQ.is_float());
    }

    #[test]
    fn awq_gptq_have_no_gguf_dtype() {
        assert_eq!(QuantScheme::AWQ.try_to_gguf_dtype(), None);
        assert_eq!(QuantScheme::GPTQ.try_to_gguf_dtype(), None);
    }
}
