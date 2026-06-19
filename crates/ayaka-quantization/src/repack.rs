//! Weight repacking: extract quants and scales from GGUF interleaved
//! super-block format into separate contiguous arrays for kernel consumption.

use half::f16;

use crate::gguf_block::{GgufDequantError, GgufDtype};
use crate::kquants;
use crate::qtensor::QTensor;
use crate::scheme::QuantScheme;

/// De-interleaved quantized weights ready for GPU upload and kernel consumption.
#[derive(Debug, Clone)]
pub struct RepackedWeights {
    /// Packed 4-bit quants, shape [out_features, in_features/2] row-major.
    /// Each byte holds 2 weights: low nibble = even index, high nibble = odd index.
    pub qs: Vec<u8>,
    /// Per-group scales as f16, shape [out_features, in_features/group_size].
    pub scales: Vec<f16>,
    /// Per-group mins as f16, shape [out_features, in_features/group_size].
    /// Present for Q4_K (which uses min+offset); empty for formats without mins.
    pub mins: Vec<f16>,
    /// Output feature dimension (N).
    pub out_features: usize,
    /// Input feature dimension (K).
    pub in_features: usize,
    /// Number of weights sharing one scale (32 for Q4_K sub-blocks).
    pub group_size: usize,
    /// Source quantization scheme.
    pub scheme: QuantScheme,
}

/// Repack quantized weights from their on-disk interleaved format into
/// separate contiguous arrays for GPU kernel consumption.
pub trait WeightRepack {
    fn repack(&self) -> Result<RepackedWeights, GgufDequantError>;
}

impl WeightRepack for QTensor {
    fn repack(&self) -> Result<RepackedWeights, GgufDequantError> {
        match self.scheme {
            QuantScheme::Q4K => repack_q4k(self),
            QuantScheme::Q6K => Err(GgufDequantError::Unsupported(GgufDtype::Q6K)),
            _ => Err(GgufDequantError::Unsupported(self.scheme.to_gguf_dtype())),
        }
    }
}

/// Extract quants, scales, and mins from Q4_K super-blocks into separate arrays.
///
/// Input: `QTensor` with `Q4K` scheme, dims `[out_features, in_features]`.
/// `bytes` holds `out_features * n_super_per_row * 144` bytes where
/// `n_super_per_row = in_features / 256`.
///
/// Output: `qs [out, in/2]`, `scales [out, in/32]`, `mins [out, in/32]`.
///
/// `in_features` must be a whole multiple of the 256-weight super-block; the
/// on-disk quant layout (sub-block `s` quants as 16 contiguous bytes with
/// low nibble = even index, high nibble = odd index) is already compatible
/// with the kernel's `[out, in/2]` row-major layout, so quants are copied
/// sub-block by sub-block while scales/mins are de-interleaved.
fn repack_q4k(qt: &QTensor) -> Result<RepackedWeights, GgufDequantError> {
    const SUPER_WEIGHTS: usize = 256;
    const SUPER_BYTES: usize = 144;
    const GROUP_SIZE: usize = 32;
    const SUBBLOCKS_PER_SUPER: usize = 8;

    if qt.dims.len() < 2 {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::Q4K,
            raw_len: qt.bytes.len(),
            expected: 0,
        });
    }
    let out_features = qt.dims[0];
    let in_features = qt.dims[1];

    if !in_features.is_multiple_of(SUPER_WEIGHTS) {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::Q4K,
            raw_len: qt.bytes.len(),
            expected: out_features * in_features.div_ceil(SUPER_WEIGHTS) * SUPER_BYTES,
        });
    }

    let n_sub_blocks = in_features / GROUP_SIZE;
    let n_super_per_row = in_features / SUPER_WEIGHTS;
    let expected = out_features * n_super_per_row * SUPER_BYTES;
    if qt.bytes.len() != expected {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::Q4K,
            raw_len: qt.bytes.len(),
            expected,
        });
    }

    let mut qs = vec![0u8; out_features * in_features / 2];
    let mut scales = vec![f16::from_f32(0.0); out_features * n_sub_blocks];
    let mut mins = vec![f16::from_f32(0.0); out_features * n_sub_blocks];

    for n in 0..out_features {
        for sb in 0..n_super_per_row {
            let super_offset = (n * n_super_per_row + sb) * SUPER_BYTES;
            let raw = &qt.bytes[super_offset..super_offset + SUPER_BYTES];

            let d = f16::from_le_bytes([raw[0], raw[1]]);
            let dmin = f16::from_le_bytes([raw[2], raw[3]]);
            let packed = &raw[4..16];
            let quant_bytes = &raw[16..SUPER_BYTES];

            for s in 0..SUBBLOCKS_PER_SUPER {
                let scale_6bit = kquants::unpack_s6(packed, s) as f32;
                let min_6bit = kquants::unpack_s6(packed, SUBBLOCKS_PER_SUPER + s) as f32;
                let scale = f16::from_f32(d.to_f32() * scale_6bit);
                let min = f16::from_f32(dmin.to_f32() * min_6bit);

                let sub_idx = sb * SUBBLOCKS_PER_SUPER + s;
                let scale_idx = n * n_sub_blocks + sub_idx;
                scales[scale_idx] = scale;
                mins[scale_idx] = min;

                let quant_offset = s * GROUP_SIZE / 2;
                let out_offset = (n * in_features + sub_idx * GROUP_SIZE) / 2;
                qs[out_offset..out_offset + GROUP_SIZE / 2]
                    .copy_from_slice(&quant_bytes[quant_offset..quant_offset + GROUP_SIZE / 2]);
            }
        }
    }

    Ok(RepackedWeights {
        qs,
        scales,
        mins,
        out_features,
        in_features,
        group_size: GROUP_SIZE,
        scheme: QuantScheme::Q4K,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufDtype;
    use half::f16;

    /// Pack 16 six-bit values into the 12-byte little-endian packed field.
    /// Inverse of `kquants::unpack_s6`.
    fn pack_s6(values: &[u8; 16]) -> [u8; 12] {
        let mut out = [0u8; 12];
        for (i, &raw_v) in values.iter().enumerate() {
            let bit_offset = i * 6;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let v = (raw_v & 0x3F) as u16;
            let word = v << bit_idx;
            out[byte_idx] |= (word & 0xFF) as u8;
            if byte_idx + 1 < 12 {
                out[byte_idx + 1] |= ((word >> 8) & 0xFF) as u8;
            }
        }
        out
    }

    /// Build a single 144-byte Q4_K super-block from explicit parameters.
    fn build_q4k_superblock(
        d: f16,
        dmin: f16,
        scales_6bit: [u8; 8],
        mins_6bit: [u8; 8],
        nibble: u8,
    ) -> [u8; 144] {
        let mut raw = [0u8; 144];
        raw[0..2].copy_from_slice(&d.to_le_bytes());
        raw[2..4].copy_from_slice(&dmin.to_le_bytes());
        let mut six_bit = [0u8; 16];
        six_bit[0..8].copy_from_slice(&scales_6bit);
        six_bit[8..16].copy_from_slice(&mins_6bit);
        raw[4..16].copy_from_slice(&pack_s6(&six_bit));
        let n = nibble & 0x0F;
        let byte = n | (n << 4);
        for b in &mut raw[16..144] {
            *b = byte;
        }
        raw
    }

    /// Dequantize a single row of a `RepackedWeights` back to f32 using the
    /// de-interleaved qs/scales/mins.
    fn manual_dequant_row(
        rw: &RepackedWeights,
        row: usize,
    ) -> Vec<f32> {
        let n_sub = rw.in_features / rw.group_size;
        let mut out = vec![0f32; rw.in_features];
        for sub in 0..n_sub {
            let scale = rw.scales[row * n_sub + sub].to_f32();
            let min = rw.mins[row * n_sub + sub].to_f32();
            for w in 0..rw.group_size {
                let bit_idx = (row * rw.in_features + sub * rw.group_size + w) / 2;
                let byte = rw.qs[bit_idx];
                let nibble = if w & 1 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                out[sub * rw.group_size + w] = scale * nibble as f32 + min;
            }
        }
        out
    }

    #[test]
    fn repack_q4k_extracts_correct_scales_and_mins() {
        // 6-bit values must be <= 63; use a sequence that stays in range.
        let d = f16::from_f32(2.0);
        let dmin = f16::from_f32(1.0);
        let scales_6bit = [10, 20, 30, 40, 50, 60, 62, 63];
        let mins_6bit = [1, 2, 3, 4, 5, 6, 7, 8];
        let nibble = 3; // byte 0x33
        let raw = build_q4k_superblock(d, dmin, scales_6bit, mins_6bit, nibble);

        let qt = QTensor::from_raw(raw.to_vec(), vec![1, 256], QuantScheme::Q4K)
            .expect("Q4K single super-block should construct");
        let rw = qt.repack().expect("Q4K repack should succeed");

        assert_eq!(rw.out_features, 1);
        assert_eq!(rw.in_features, 256);
        assert_eq!(rw.group_size, 32);
        assert_eq!(rw.scheme, QuantScheme::Q4K);

        // qs: 1 * 256 / 2 = 128 bytes, all 0x33 (low=high=3).
        assert_eq!(rw.qs.len(), 128);
        assert!(
            rw.qs.iter().all(|&b| b == 0x33),
            "all qs bytes should be 0x33"
        );

        // scales/mins: 1 * 256/32 = 8 entries each.
        assert_eq!(rw.scales.len(), 8);
        assert_eq!(rw.mins.len(), 8);
        assert_eq!(rw.scales[0].to_f32(), 2.0 * 10.0);
        assert_eq!(rw.scales[7].to_f32(), 2.0 * 63.0);
        assert_eq!(rw.mins[0].to_f32(), 1.0 * 1.0);
        assert_eq!(rw.mins[7].to_f32(), 1.0 * 8.0);
    }

    #[test]
    fn repack_q4k_preserves_dequant_values() {
        // Choose values whose d*scale and dmin*min products are exactly
        // representable in f16 (integers <= 2048), so the repacked f16
        // scales/mins round-trip exactly and match the on-disk dequant.
        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(1.0);
        let scales_6bit = [2, 3, 4, 5, 6, 7, 8, 9];
        let mins_6bit = [1, 1, 1, 1, 1, 1, 1, 1];
        let nibble = 5; // byte 0x55
        let raw = build_q4k_superblock(d, dmin, scales_6bit, mins_6bit, nibble);

        let qt = QTensor::from_raw(raw.to_vec(), vec![1, 256], QuantScheme::Q4K)
            .expect("Q4K single super-block should construct");
        let rw = qt.repack().expect("Q4K repack should succeed");

        let manual = manual_dequant_row(&rw, 0);
        let reference = GgufDtype::Q4K
            .dequantize(&raw, 256)
            .expect("on-disk dequant should succeed");
        assert_eq!(manual.len(), 256);
        assert_eq!(reference.len(), 256);
        assert_eq!(manual, reference);
    }

    #[test]
    fn repack_q4k_multi_row() {
        // dims [2, 512]: 2 rows, 2 super-blocks per row, 4 super-blocks total.
        let row0_sb0 = build_q4k_superblock(
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            [2; 8],
            [1; 8],
            1, // 0x11
        );
        let row0_sb1 = build_q4k_superblock(
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            [2; 8],
            [1; 8],
            1, // 0x11
        );
        let row1_sb0 = build_q4k_superblock(
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            [3; 8],
            [5; 8],
            2, // 0x22
        );
        let row1_sb1 = build_q4k_superblock(
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            [3; 8],
            [5; 8],
            2, // 0x22
        );

        let mut bytes = Vec::with_capacity(4 * 144);
        bytes.extend_from_slice(&row0_sb0);
        bytes.extend_from_slice(&row0_sb1);
        bytes.extend_from_slice(&row1_sb0);
        bytes.extend_from_slice(&row1_sb1);

        let qt = QTensor::from_raw(bytes, vec![2, 512], QuantScheme::Q4K)
            .expect("Q4K 4 super-blocks should construct");
        let rw = qt
            .repack()
            .expect("Q4K multi-row repack should succeed");

        // qs: 2 * 512 / 2 = 512 bytes; scales/mins: 2 * 512/32 = 32 entries.
        assert_eq!(rw.qs.len(), 512);
        assert_eq!(rw.scales.len(), 32);
        assert_eq!(rw.mins.len(), 32);

        // Row 0 quants occupy bytes 0..256, row 1 occupies 256..512.
        assert!(rw.qs[0..256].iter().all(|&b| b == 0x11));
        assert!(rw.qs[256..512].iter().all(|&b| b == 0x22));

        // Row 0 scales (idx 0..16) = 1.0 * 2 = 2.0; row 1 (idx 16..32) = 2.0 * 3 = 6.0.
        assert!(rw.scales[0..16].iter().all(|s| s.to_f32() == 2.0));
        assert!(
            rw.scales[16..32]
                .iter()
                .all(|s| s.to_f32() == 6.0)
        );

        // Row 0 mins (idx 0..16) = 1.0 * 1 = 1.0; row 1 (idx 16..32) = 1.0 * 5 = 5.0.
        assert!(rw.mins[0..16].iter().all(|m| m.to_f32() == 1.0));
        assert!(rw.mins[16..32].iter().all(|m| m.to_f32() == 5.0));
    }

    #[test]
    fn repack_rejects_unsupported_scheme() {
        // Q6K: 1 super-block = 210 bytes for 256 weights.
        let qt = QTensor::from_raw(vec![0u8; 210], vec![1, 256], QuantScheme::Q6K)
            .expect("Q6K single super-block should construct");
        let err = qt.repack().unwrap_err();
        assert_eq!(err, GgufDequantError::Unsupported(GgufDtype::Q6K));
    }
}
