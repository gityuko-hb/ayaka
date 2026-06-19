//! GGUF tensor dtypes, their block layouts, and dequantization to `f32`.
//!
//! Quantized GGUF blocks must be dequantized to a float layout before any
//! kernel sees them (kernels only accept F32/F16/BF16). `ayaka-loader` casts
//! the `f32` output here down to F16/BF16 as needed.

use half::{bf16, f16};

use crate::scheme::QuantScheme;

/// Numeric type id of a GGUF tensor, matching the upstream `ggml` enum values.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GgufDtype {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
    MXFP4,
}

/// Block geometry of a GGUF dtype: how many weights share one stored block and
/// how many bytes that block occupies on disk.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GgufBlock {
    /// Number of scalar weights per stored block.
    pub weights_per_block: usize,
    /// Size of one stored block in bytes (quants + scales).
    pub bytes_per_block: usize,
}

/// Error raised when a GGUF dtype cannot yet be dequantized or is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufDequantError {
    /// The dtype's dequant path is not implemented yet.
    Unsupported(GgufDtype),
    /// `raw.len()` is not a whole number of blocks for `n_weights`.
    MalformedLength {
        dtype: GgufDtype,
        raw_len: usize,
        expected: usize,
    },
}

impl core::fmt::Display for GgufDequantError {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            GgufDequantError::Unsupported(d) => {
                write!(f, "GGUF dequant not implemented for {d:?}")
            },
            GgufDequantError::MalformedLength {
                dtype,
                raw_len,
                expected,
            } => write!(
                f,
                "GGUF {dtype:?} buffer is {raw_len} bytes, expected {expected}"
            ),
        }
    }
}

impl core::error::Error for GgufDequantError {}

impl GgufDtype {
    /// Map a raw `ggml` type id to a [`GgufDtype`], if recognized.
    pub const fn from_ggml_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(GgufDtype::F32),
            1 => Some(GgufDtype::F16),
            2 => Some(GgufDtype::Q4_0),
            8 => Some(GgufDtype::Q8_0),
            12 => Some(GgufDtype::Q4K),
            14 => Some(GgufDtype::Q6K),
            30 => Some(GgufDtype::BF16),
            39 => Some(GgufDtype::MXFP4),
            _ => None,
        }
    }

    /// Block geometry of this dtype.
    pub const fn block(self) -> GgufBlock {
        match self {
            GgufDtype::F32 => GgufBlock {
                weights_per_block: 1,
                bytes_per_block: 4,
            },
            GgufDtype::F16 | GgufDtype::BF16 => GgufBlock {
                weights_per_block: 1,
                bytes_per_block: 2,
            },
            GgufDtype::Q4_0 => GgufBlock {
                weights_per_block: 32,
                bytes_per_block: 18,
            },
            GgufDtype::Q8_0 => GgufBlock {
                weights_per_block: 32,
                bytes_per_block: 34,
            },
            GgufDtype::Q4K => GgufBlock {
                weights_per_block: 256,
                bytes_per_block: 144,
            },
            GgufDtype::Q6K => GgufBlock {
                weights_per_block: 256,
                bytes_per_block: 210,
            },
            GgufDtype::MXFP4 => GgufBlock {
                weights_per_block: 32,
                bytes_per_block: 17,
            },
        }
    }

    /// The [`QuantScheme`] this dtype corresponds to.
    pub const fn scheme(self) -> QuantScheme {
        match self {
            GgufDtype::F32 | GgufDtype::F16 => QuantScheme::F16,
            GgufDtype::BF16 => QuantScheme::BF16,
            GgufDtype::Q4_0 => QuantScheme::Q4_0,
            GgufDtype::Q8_0 => QuantScheme::Q8_0,
            GgufDtype::Q4K => QuantScheme::Q4K,
            GgufDtype::Q6K => QuantScheme::Q6K,
            GgufDtype::MXFP4 => QuantScheme::MXFP4,
        }
    }

    /// Dequantize `n_weights` scalars from `raw` block bytes into `f32`.
    ///
    /// Implemented for F32/F16/BF16/Q4_0/Q8_0. The K-quants and MXFP4 return
    /// [`GgufDequantError::Unsupported`] (future work — keep-quantized and
    /// in-kernel dequant are out of scope).
    pub fn dequantize(
        self,
        raw: &[u8],
        n_weights: usize,
    ) -> Result<Vec<f32>, GgufDequantError> {
        let block = self.block();
        let n_blocks = n_weights.div_ceil(block.weights_per_block);
        let expected = n_blocks * block.bytes_per_block;
        if raw.len() < expected {
            return Err(GgufDequantError::MalformedLength {
                dtype: self,
                raw_len: raw.len(),
                expected,
            });
        }
        match self {
            GgufDtype::F32 => Ok(raw
                .chunks_exact(4)
                .take(n_weights)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            GgufDtype::F16 => Ok(raw
                .chunks_exact(2)
                .take(n_weights)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            GgufDtype::BF16 => Ok(raw
                .chunks_exact(2)
                .take(n_weights)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            GgufDtype::Q8_0 => Ok(dequantize_q8_0(raw, n_weights)),
            GgufDtype::Q4_0 => Ok(dequantize_q4_0(raw, n_weights)),
            GgufDtype::Q4K | GgufDtype::Q6K | GgufDtype::MXFP4 => {
                Err(GgufDequantError::Unsupported(self))
            },
        }
    }
}

/// Q8_0: each 34-byte block is an f16 scale `d` followed by 32 `i8` quants;
/// `weight = d * q`.
fn dequantize_q8_0(
    raw: &[u8],
    n_weights: usize,
) -> Vec<f32> {
    const QK: usize = 32;
    let mut out = Vec::with_capacity(n_weights);
    for block in raw.chunks_exact(34) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..2 + QK] {
            out.push(d * (q as i8) as f32);
        }
    }
    out.truncate(n_weights);
    out
}

/// Q4_0: each 18-byte block is an f16 scale `d` followed by 16 bytes packing 32
/// 4-bit quants; low nibbles fill weights `0..16`, high nibbles `16..32`, each
/// offset by 8: `weight = d * (nibble - 8)`.
fn dequantize_q4_0(
    raw: &[u8],
    n_weights: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_weights);
    for block in raw.chunks_exact(18) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let mut lo = [0f32; 16];
        let mut hi = [0f32; 16];
        for (j, &byte) in qs.iter().enumerate() {
            lo[j] = d * (((byte & 0x0F) as i32) - 8) as f32;
            hi[j] = d * (((byte >> 4) as i32) - 8) as f32;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out.truncate(n_weights);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_ids_round_trip_known_types() {
        assert_eq!(GgufDtype::from_ggml_id(0), Some(GgufDtype::F32));
        assert_eq!(GgufDtype::from_ggml_id(2), Some(GgufDtype::Q4_0));
        assert_eq!(GgufDtype::from_ggml_id(8), Some(GgufDtype::Q8_0));
        assert_eq!(GgufDtype::from_ggml_id(14), Some(GgufDtype::Q6K));
        assert_eq!(GgufDtype::from_ggml_id(30), Some(GgufDtype::BF16));
        assert_eq!(GgufDtype::from_ggml_id(999), None);
    }

    #[test]
    fn q6k_block_geometry() {
        assert_eq!(
            GgufDtype::Q6K.block(),
            GgufBlock {
                weights_per_block: 256,
                bytes_per_block: 210,
            }
        );
    }

    #[test]
    fn q6k_ggml_id_round_trip() {
        assert_eq!(GgufDtype::from_ggml_id(14), Some(GgufDtype::Q6K));
    }

    #[test]
    fn q6k_scheme_mapping() {
        assert_eq!(GgufDtype::Q6K.scheme(), QuantScheme::Q6K);
    }

    #[test]
    fn block_geometry_matches_bytes_per_weight() {
        for d in [GgufDtype::Q4_0, GgufDtype::Q8_0, GgufDtype::MXFP4] {
            let b = d.block();
            let bpw = b.bytes_per_block as f64 / b.weights_per_block as f64;
            assert_eq!(bpw, d.scheme().bytes_per_weight());
        }
    }

    #[test]
    fn dequant_q8_0_recovers_scaled_quants() {
        // One block: scale = 0.5, quants = [-2, 0, 3, ...].
        let d = f16::from_f32(0.5);
        let mut raw = Vec::new();
        raw.extend_from_slice(&d.to_le_bytes());
        let mut quants = [0i8; 32];
        quants[0] = -2;
        quants[2] = 3;
        for &q in &quants {
            raw.push(q as u8);
        }
        let out = GgufDtype::Q8_0.dequantize(&raw, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], -1.0); // 0.5 * -2
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.5); // 0.5 * 3
    }

    #[test]
    fn dequant_q4_0_applies_offset_and_split() {
        // scale = 1.0; first byte low nibble = 0x0 -> -8, high nibble = 0xF -> 7.
        let d = f16::from_f32(1.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&d.to_le_bytes());
        raw.push(0xF0); // low=0 -> -8, high=15 -> 7
        raw.extend_from_slice(&[0u8; 15]); // rest -> -8
        let out = GgufDtype::Q4_0.dequantize(&raw, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], -8.0); // weight 0 from low nibble of byte 0
        assert_eq!(out[16], 7.0); // weight 16 from high nibble of byte 0
        assert_eq!(out[1], -8.0); // low nibble of byte 1 (0x0)
    }

    #[test]
    fn dequant_rejects_short_buffer() {
        let err = GgufDtype::Q8_0
            .dequantize(&[0u8; 10], 32)
            .unwrap_err();
        assert!(matches!(err, GgufDequantError::MalformedLength { .. }));
    }

    #[test]
    fn kquants_unsupported_for_now() {
        let raw = vec![0u8; 144];
        let err = GgufDtype::Q4K.dequantize(&raw, 256).unwrap_err();
        assert_eq!(err, GgufDequantError::Unsupported(GgufDtype::Q4K));

        let raw = vec![0u8; 210];
        let err = GgufDtype::Q6K.dequantize(&raw, 256).unwrap_err();
        assert_eq!(err, GgufDequantError::Unsupported(GgufDtype::Q6K));
    }
}
