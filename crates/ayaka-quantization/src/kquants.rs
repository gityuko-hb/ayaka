//! CPU reference dequantization for the GGUF "K-quants" (Q4_K, Q6_K).
//!
//! These are super-block formats (256 weights per super-block) with nested
//! per-sub-block scales. The functions here are the reference path used by
//! [`crate::GgufDtype::dequantize`] before any kernel-accelerated dequant
//! exists; kernels always receive F32/F16/BF16.

use crate::{GgufDequantError, GgufDtype};
use half::f16;

/// Q4_K super-block size on disk: 256 weights packed into 144 bytes.
const Q4K_BLOCK_BYTES: usize = 144;
/// Number of weights in one Q4_K / Q6_K super-block.
const QK_K: usize = 256;
/// Number of sub-blocks inside one K-quant super-block.
const QK_K_SUBBLOCKS: usize = 8;
/// Weights per K-quant sub-block.
const QK_K_SUBBLOCK_SIZE: usize = 32;

/// Dequantize a Q4_K tensor to `f32`.
///
/// Each 144-byte super-block holds an f16 `d`, an f16 `dmin`, 16 six-bit values
/// (8 sub-block scales followed by 8 sub-block mins, packed into 12 bytes), and
/// 128 bytes of 4-bit quants (256 weights). For sub-block `s` and weight `w`:
/// `output = d * scales[s] * nibble + dmin * mins[s]`.
///
/// `raw` must contain at least `n_blocks * 144` bytes where
/// `n_blocks = n_weights.div_ceil(256)`; otherwise a [`GgufDequantError::MalformedLength`]
/// is returned. The output is truncated to exactly `n_weights`.
pub fn dequantize_q4_k(
    raw: &[u8],
    n_weights: usize,
) -> Result<Vec<f32>, GgufDequantError> {
    let block = GgufDtype::Q4K.block();
    let n_blocks = n_weights.div_ceil(block.weights_per_block);
    let expected = n_blocks * block.bytes_per_block;
    if raw.len() < expected {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::Q4K,
            raw_len: raw.len(),
            expected,
        });
    }

    let mut out = Vec::with_capacity(n_weights);
    for super_block in raw.chunks_exact(Q4K_BLOCK_BYTES) {
        let d = f16::from_le_bytes([super_block[0], super_block[1]]).to_f32();
        let dmin = f16::from_le_bytes([super_block[2], super_block[3]]).to_f32();
        let scales_bytes = &super_block[4..16];
        let mut scales = [0u8; QK_K_SUBBLOCKS];
        let mut mins = [0u8; QK_K_SUBBLOCKS];
        for i in 0..QK_K_SUBBLOCKS {
            scales[i] = unpack_s6(scales_bytes, i);
            mins[i] = unpack_s6(scales_bytes, QK_K_SUBBLOCKS + i);
        }
        let qs = &super_block[16..Q4K_BLOCK_BYTES];
        let mut block_out = [0f32; QK_K];
        for s in 0..QK_K_SUBBLOCKS {
            let scale_s = d * scales[s] as f32;
            let min_s = dmin * mins[s] as f32;
            for w in 0..QK_K_SUBBLOCK_SIZE {
                let byte = qs[s * 16 + w / 2];
                let nibble = if w & 1 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                block_out[s * QK_K_SUBBLOCK_SIZE + w] = scale_s * nibble as f32 + min_s;
            }
        }
        out.extend_from_slice(&block_out);
    }
    out.truncate(n_weights);
    Ok(out)
}

/// Dequantize a Q6_K tensor to `f32`.
///
/// Each 210-byte super-block holds 128 bytes of lower 4-bit quants (`ql`), 64
/// bytes of upper 2-bit quants (`qh`), 8 int8 sub-block scales (in a 16-byte
/// field), and an f16 super-block scale `d` at offset 208. For weight `n` in
/// sub-block `n / 32`: `q6 = ql4 | (qh2 << 4)` (0..63), and
/// `output = d * scales[sub_block] * (q6 - 32)`.
///
/// `raw` must contain at least `n_blocks * 210` bytes where
/// `n_blocks = n_weights.div_ceil(256)`; otherwise a [`GgufDequantError::MalformedLength`]
/// is returned. The output is truncated to exactly `n_weights`.
pub fn dequantize_q6_k(
    raw: &[u8],
    n_weights: usize,
) -> Result<Vec<f32>, GgufDequantError> {
    let block = GgufDtype::Q6K.block();
    let n_blocks = n_weights.div_ceil(block.weights_per_block);
    let expected = n_blocks * block.bytes_per_block;
    if raw.len() < expected {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::Q6K,
            raw_len: raw.len(),
            expected,
        });
    }

    let mut out = Vec::with_capacity(n_weights);
    for super_block in raw.chunks_exact(block.bytes_per_block) {
        let d = f16::from_le_bytes([super_block[208], super_block[209]]).to_f32();
        let mut scales = [0i8; QK_K_SUBBLOCKS];
        for i in 0..QK_K_SUBBLOCKS {
            scales[i] = super_block[192 + i] as i8;
        }
        let ql = &super_block[0..128];
        let qh = &super_block[128..192];
        let mut block_out = [0f32; QK_K];
        for n in 0..QK_K {
            let sub_block = n / QK_K_SUBBLOCK_SIZE;
            let q4 = (ql[n / 2] >> ((n % 2) * 4)) & 0x0F;
            let q2 = (qh[n / 4] >> ((n % 4) * 2)) & 0x03;
            let q6 = (q4 | (q2 << 4)) as i32;
            block_out[n] = d * scales[sub_block] as f32 * (q6 - 32) as f32;
        }
        out.extend_from_slice(&block_out);
    }
    out.truncate(n_weights);
    Ok(out)
}

/// Unpack the `i`-th six-bit value (0..15) from a 12-byte packed scales field.
///
/// The 16 six-bit values are laid out little-endian across the 96-bit field:
/// value `i` starts at bit `i * 6`. Values `0..8` are the sub-block scales and
/// `8..16` are the sub-block mins.
pub(crate) fn unpack_s6(
    scales: &[u8],
    i: usize,
) -> u8 {
    let bit_offset = i * 6;
    let byte_idx = bit_offset / 8;
    let bit_idx = bit_offset % 8;
    if bit_idx <= 2 {
        (scales[byte_idx] >> bit_idx) & 0x3F
    } else {
        let lo = (scales[byte_idx] >> bit_idx) & 0x3F;
        let hi = (scales[byte_idx + 1] << (8 - bit_idx)) & 0x3F;
        lo | hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufDequantError;
    use half::f16;

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

    #[test]
    fn dequant_q4k_recovers_superblock() {
        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(0.5);
        let mut raw = [0u8; 144];
        raw[0..2].copy_from_slice(&d.to_le_bytes());
        raw[2..4].copy_from_slice(&dmin.to_le_bytes());

        let mut six_bit = [0u8; 16];
        six_bit[0..8].copy_from_slice(&[2, 3, 4, 5, 6, 7, 8, 9]);
        six_bit[8..16].copy_from_slice(&[1; 8]);
        raw[4..16].copy_from_slice(&pack_s6(&six_bit));

        for b in &mut raw[16..144] {
            *b = 0x55;
        }

        let out = dequantize_q4_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 10.5);
        assert_eq!(out[31], 10.5);
        assert_eq!(out[32], 15.5);
        assert_eq!(out[224], 45.5);
    }

    #[test]
    fn dequant_q6k_recovers_superblock() {
        let d = f16::from_f32(2.0);
        let mut raw = [0u8; 210];
        for b in &mut raw[0..128] {
            *b = 0x55;
        }
        raw[192..200].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        raw[208..210].copy_from_slice(&d.to_le_bytes());

        let out = dequantize_q6_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -54.0);
        assert_eq!(out[31], -54.0);
        assert_eq!(out[32], -108.0);
        assert_eq!(out[224], -432.0);
    }

    #[test]
    fn dequant_q4k_rejects_short_buffer() {
        let err = dequantize_q4_k(&[0u8; 10], 256).unwrap_err();
        assert!(matches!(err, GgufDequantError::MalformedLength { .. }));
    }

    #[test]
    fn dequant_q6k_rejects_short_buffer() {
        let err = dequantize_q6_k(&[0u8; 10], 256).unwrap_err();
        assert!(matches!(err, GgufDequantError::MalformedLength { .. }));
    }
}
