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
            _ => Err(GgufDequantError::Unsupported(
                self.scheme
                    .try_to_gguf_dtype()
                    .unwrap_or(GgufDtype::F32),
            )),
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

/// Input for AWQ/GPTQ repack: three separate tensors from HuggingFace safetensors.
///
/// HuggingFace AWQ/GPTQ stores quantized weights as three sibling tensors per
/// linear layer: `qweight` (int32-packed int4 weights), `qzeros` (int32-packed
/// zero-points, empty for symmetric), and `scales` (f16 per-group scales). This
/// struct collects them plus the layer geometry and the source scheme.
#[derive(Debug, Clone)]
pub struct AwqGptqInput {
    /// Packed int4 weights as int32 values, shape `[K/8, N]` (column-major).
    /// Each int32 holds 8 int4 weights packed along the K dimension
    /// (nibble `i` → K position `k_block*8 + i`).
    pub qweight: Vec<u32>,
    /// Packed int4 zero-points as int32 values, shape `[K/group/8, N]`.
    /// Each int32 holds 8 zero-points. Empty for symmetric quantization
    /// (in which case a default zero of 8 is used).
    pub qzeros: Vec<u32>,
    /// Per-group scales as f16, shape `[N, K/group]` (row-major).
    pub scales: Vec<f16>,
    /// Output feature dimension (N).
    pub out_features: usize,
    /// Input feature dimension (K).
    pub in_features: usize,
    /// Number of weights sharing one scale/zero.
    pub group_size: usize,
    /// Source quantization scheme (AWQ or GPTQ).
    pub scheme: QuantScheme,
}

/// Repack AWQ/GPTQ weights into the kernel-friendly [`RepackedWeights`] format.
///
/// Converts from `[K/8, N]` int32-packed column-major to `[N, K/2]` U8-packed
/// row-major. Computes `mins = -scale * zero` so the kernel's
/// `scale * nibble + min` formula yields `scale * (nibble - zero)`, matching
/// the HuggingFace dequant convention without changing the C ABI.
///
/// For symmetric quantization (`qzeros` empty), a default zero of 8 is used,
/// which maps unsigned int4 nibbles to signed weights via `scale * (nibble - 8)`.
pub fn repack_awq_gptq(input: &AwqGptqInput) -> Result<RepackedWeights, GgufDequantError> {
    let n = input.out_features;
    let k = input.in_features;
    let group_size = input.group_size;

    if k == 0 || n == 0 || group_size == 0 || !k.is_multiple_of(8) || !k.is_multiple_of(group_size)
    {
        return Err(GgufDequantError::Unsupported(
            input
                .scheme
                .try_to_gguf_dtype()
                .unwrap_or(GgufDtype::F32),
        ));
    }
    let n_groups = k / group_size;

    if input.qweight.len() != (k / 8) * n {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::F32,
            raw_len: input.qweight.len() * 4,
            expected: (k / 8) * n * 4,
        });
    }
    if input.scales.len() != n * n_groups {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::F32,
            raw_len: input.scales.len() * 2,
            expected: n * n_groups * 2,
        });
    }

    let has_zeros = !input.qzeros.is_empty();
    if has_zeros && input.qzeros.len() != (n_groups / 8) * n {
        return Err(GgufDequantError::MalformedLength {
            dtype: GgufDtype::F32,
            raw_len: input.qzeros.len() * 4,
            expected: (n_groups / 8) * n * 4,
        });
    }
    if has_zeros && !n_groups.is_multiple_of(8) {
        return Err(GgufDequantError::Unsupported(
            input
                .scheme
                .try_to_gguf_dtype()
                .unwrap_or(GgufDtype::F32),
        ));
    }

    let mut qs = vec![0u8; n * k / 2];
    let mut scales_out = vec![f16::from_f32(0.0); n * n_groups];
    let mut mins_out = vec![f16::from_f32(0.0); n * n_groups];

    for col_n in 0..n {
        for g in 0..n_groups {
            let scale = input.scales[col_n * n_groups + g];
            scales_out[col_n * n_groups + g] = scale;

            // Zero-point for group g: qzeros[g/8, col_n], nibble (g % 8).
            // Symmetric default is 8, mapping unsigned nibble to signed weight.
            let zero = if has_zeros {
                let qz_int32 = input.qzeros[(g / 8) * n + col_n];
                let z_nibble = (qz_int32 >> ((g % 8) * 4)) & 0xF;
                z_nibble as f32
            } else {
                8.0
            };

            // mins = -scale * zero → kernel's scale*nibble + min = scale*(nibble - zero).
            mins_out[col_n * n_groups + g] = f16::from_f32(-scale.to_f32() * zero);
        }

        // Unpack qweight from [K/8, N] int32 to [N, K/2] U8.
        // qweight[k_block, col_n] packs 8 nibbles for K = k_block*8 .. k_block*8+7.
        // Output byte (k_block*4 + i) holds weight (2i) in low nibble and
        // weight (2i+1) in high nibble, matching the kernel's row-major layout.
        for k_block in 0..k / 8 {
            let packed = input.qweight[k_block * n + col_n];
            for i in 0..4 {
                let nibble_lo = (packed >> (i * 8)) & 0xF;
                let nibble_hi = (packed >> (i * 8 + 4)) & 0xF;
                let byte = (nibble_lo as u8) | ((nibble_hi as u8) << 4);
                let k_byte = k_block * 4 + i;
                qs[col_n * (k / 2) + k_byte] = byte;
            }
        }
    }

    Ok(RepackedWeights {
        qs,
        scales: scales_out,
        mins: mins_out,
        out_features: n,
        in_features: k,
        group_size,
        scheme: input.scheme,
    })
}

/// Repack `RepackedWeights.qs` from `[N, K/2]` row-major into Marlin tile
/// layout `[N/16, K/16, 16, 8]` for Tensor Core `mma.sync.m16n8k16` access.
///
/// Each 16x16 weight tile is stored as 16 rows of 8 bytes (16 nibbles per
/// row). Scales and mins are passed through unchanged because the kernel
/// indexes them by group, not by tile. Only `qs` is shuffled.
///
/// `out_features` (N) and `in_features` (K) must both be multiples of 16;
/// otherwise the layout cannot be tiled and [`GgufDequantError::Unsupported`]
/// is returned.
pub fn repack_to_marlin(rw: &RepackedWeights) -> Result<RepackedWeights, GgufDequantError> {
    let n = rw.out_features;
    let k = rw.in_features;

    // N and K must be multiples of 16 for tiling.
    if !n.is_multiple_of(16) || !k.is_multiple_of(16) {
        return Err(GgufDequantError::Unsupported(
            rw.scheme
                .try_to_gguf_dtype()
                .unwrap_or(GgufDtype::F32),
        ));
    }

    let n_tiles_n = n / 16;
    let n_tiles_k = k / 16;
    let mut shuffled = vec![0u8; rw.qs.len()]; // same total size

    for tile_n in 0..n_tiles_n {
        for tile_k in 0..n_tiles_k {
            // Copy 16 rows x 8 bytes (16 nibbles per row = 16 weights per row).
            for row in 0..16 {
                let src_n = tile_n * 16 + row;
                let src_k_byte = tile_k * 8; // 16 weights = 8 bytes
                let src_offset = src_n * (k / 2) + src_k_byte;

                let dst_tile = tile_n * n_tiles_k + tile_k;
                let dst_offset = dst_tile * 16 * 8 + row * 8;

                shuffled[dst_offset..dst_offset + 8]
                    .copy_from_slice(&rw.qs[src_offset..src_offset + 8]);
            }
        }
    }

    Ok(RepackedWeights {
        qs: shuffled,
        scales: rw.scales.clone(),
        mins: rw.mins.clone(),
        out_features: n,
        in_features: k,
        group_size: rw.group_size,
        scheme: rw.scheme,
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

    /// Dequantize a single row reading directly from a Marlin-shuffled
    /// `RepackedWeights` (layout `[N/16, K/16, 16, 8]`). This mirrors the
    /// kernel's intended tile access pattern, so it tests that the shuffle is
    /// consumable as designed rather than merely a size-preserving copy.
    fn marlin_dequant_row(
        rw: &RepackedWeights,
        row: usize,
    ) -> Vec<f32> {
        assert!(row < rw.out_features);
        let k = rw.in_features;
        let n_tiles_k = k / 16;
        let n_sub = k / rw.group_size;
        let mut out = vec![0f32; k];
        let tile_n = row / 16;
        let row_in_tile = row % 16;
        for tk in 0..n_tiles_k {
            let dst_tile = tile_n * n_tiles_k + tk;
            let tile_base = dst_tile * 16 * 8; // 128 bytes per 16x16 tile
            for k_in in 0..16 {
                let k_global = tk * 16 + k_in;
                let byte = rw.qs[tile_base + row_in_tile * 8 + k_in / 2];
                let nibble = if k_in & 1 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let group_idx = k_global / rw.group_size;
                let scale = rw.scales[row * n_sub + group_idx].to_f32();
                let min = rw.mins[row * n_sub + group_idx].to_f32();
                out[k_global] = scale * nibble as f32 + min;
            }
        }
        out
    }

    #[test]
    fn repack_to_marlin_preserves_dequant_values() {
        // 16 rows x 256 cols: one N-tile, 16 K-tiles. Each row carries a
        // unique (scale, min, nibble) signature so any shuffle permutation
        // error (row/column/tile mixup) changes the dequant values.
        let mut bytes = Vec::with_capacity(16 * 144);
        for row in 0..16u8 {
            let raw = build_q4k_superblock(
                f16::from_f32(1.0),
                f16::from_f32(1.0),
                [2 * row + 1; 8], // 1,3,5,...,31 (all <= 63)
                [row + 1; 8],     // 1..16 (all <= 63)
                row,              // nibble == row index (0..15)
            );
            bytes.extend_from_slice(&raw);
        }
        let qt = QTensor::from_raw(bytes, vec![16, 256], QuantScheme::Q4K)
            .expect("Q4K 16-row tensor should construct");
        let rw = qt.repack().expect("Q4K repack should succeed");

        // Reference dequant from the unshuffled [N, K/2] layout.
        let reference: Vec<Vec<f32>> = (0..16)
            .map(|r| manual_dequant_row(&rw, r))
            .collect();

        // Shuffle to Marlin tile layout.
        let marlin = repack_to_marlin(&rw).expect("marlin repack should succeed");

        // Size preserved; scales/mins passed through unchanged.
        assert_eq!(marlin.qs.len(), rw.qs.len());
        assert_eq!(marlin.scales, rw.scales);
        assert_eq!(marlin.mins, rw.mins);
        assert_eq!(marlin.out_features, 16);
        assert_eq!(marlin.in_features, 256);

        // Dequant reading through the Marlin access pattern must match.
        for (r, expected) in reference.iter().enumerate() {
            let got = marlin_dequant_row(&marlin, r);
            assert_eq!(
                got, *expected,
                "row {r} dequant mismatch after marlin shuffle"
            );
        }
    }

    #[test]
    fn repack_to_marlin_rejects_non_tile_aligned_shape() {
        // N=8 is not a multiple of 16 -> tiling is impossible.
        let rw = RepackedWeights {
            qs: vec![0u8; 8 * 16 / 2],
            scales: vec![f16::from_f32(1.0); 8],
            mins: vec![f16::from_f32(0.0); 8],
            out_features: 8,
            in_features: 16,
            group_size: 16,
            scheme: QuantScheme::Q4K,
        };
        let err = repack_to_marlin(&rw).unwrap_err();
        assert_eq!(
            err,
            GgufDequantError::Unsupported(QuantScheme::Q4K.try_to_gguf_dtype().unwrap())
        );
    }

    /// Dequantize one row of a `RepackedWeights` using the kernel formula
    /// `scale * nibble + min`. Reused for AWQ/GPTQ where `min = -scale * zero`.
    fn awq_dequant_row(
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
    fn repack_awq_gptq_nibble_packing_is_correct() {
        // N=1, K=8, group_size=8, n_groups=1. One int32 packs 8 nibbles 0..7.
        // packed = 0x76543210 (nibble 0 = 0, ..., nibble 7 = 7).
        let input = AwqGptqInput {
            qweight: vec![0x76543210],
            qzeros: vec![],
            scales: vec![f16::from_f32(1.0)],
            out_features: 1,
            in_features: 8,
            group_size: 8,
            scheme: QuantScheme::AWQ,
        };
        let rw = repack_awq_gptq(&input).expect("repack should succeed");

        // qs: [1, 4] = 4 bytes. byte i holds nibble(2i) low, nibble(2i+1) high.
        assert_eq!(rw.qs.len(), 4);
        assert_eq!(rw.qs[0], 0x10); // nibble 0 | nibble 1 << 4
        assert_eq!(rw.qs[1], 0x32); // nibble 2 | nibble 3 << 4
        assert_eq!(rw.qs[2], 0x54); // nibble 4 | nibble 5 << 4
        assert_eq!(rw.qs[3], 0x76); // nibble 6 | nibble 7 << 4
    }

    #[test]
    fn repack_awq_gptq_preserves_dequant_values_asymmetric() {
        // N=2, K=32, group_size=4, n_groups=8.
        // scales = 2.0, zeros = 5, nibbles = 3 → weight = 2*(3-5) = -4.0.
        let n = 2;
        let k = 32;
        let group_size = 4;
        let n_groups = k / group_size; // 8

        let qweight = vec![0x33333333u32; (k / 8) * n]; // 8 int32s
        // qzeros: [n_groups/8, N] = [1, 2] = 2 int32s, all nibbles = 5.
        let qzeros = vec![0x55555555u32; (n_groups / 8) * n];
        let scales = vec![f16::from_f32(2.0); n * n_groups];

        let input = AwqGptqInput {
            qweight,
            qzeros,
            scales,
            out_features: n,
            in_features: k,
            group_size,
            scheme: QuantScheme::GPTQ,
        };
        let rw = repack_awq_gptq(&input).expect("repack should succeed");

        assert_eq!(rw.out_features, n);
        assert_eq!(rw.in_features, k);
        assert_eq!(rw.group_size, group_size);
        assert_eq!(rw.scheme, QuantScheme::GPTQ);
        assert_eq!(rw.qs.len(), n * k / 2);
        assert_eq!(rw.scales.len(), n * n_groups);
        assert_eq!(rw.mins.len(), n * n_groups);

        // All scales = 2.0, all mins = -2.0 * 5 = -10.0.
        assert!(rw.scales.iter().all(|s| s.to_f32() == 2.0));
        assert!(rw.mins.iter().all(|m| m.to_f32() == -10.0));

        // Dequant: 2.0 * 3 + (-10.0) = -4.0 for every weight.
        for row in 0..n {
            let deq = awq_dequant_row(&rw, row);
            assert_eq!(deq.len(), k);
            assert!(deq.iter().all(|v| (*v - (-4.0)).abs() < 1e-3), "row {row}");
        }
    }

    #[test]
    fn repack_awq_gptq_symmetric_uses_default_zero() {
        // Symmetric: qzeros empty → zero defaults to 8.
        // scale = 2.0, nibble = 3 → weight = 2*(3-8) = -10.0, min = -16.0.
        let n = 1;
        let k = 16;
        let group_size = 8;
        let n_groups = k / group_size; // 2

        let input = AwqGptqInput {
            qweight: vec![0x33333333u32; (k / 8) * n],
            qzeros: vec![],
            scales: vec![f16::from_f32(2.0); n * n_groups],
            out_features: n,
            in_features: k,
            group_size,
            scheme: QuantScheme::AWQ,
        };
        let rw = repack_awq_gptq(&input).expect("repack should succeed");

        // mins = -2.0 * 8 = -16.0.
        assert!(rw.mins.iter().all(|m| m.to_f32() == -16.0));

        let deq = awq_dequant_row(&rw, 0);
        // 2.0 * 3 + (-16.0) = -10.0.
        assert!(deq.iter().all(|v| (*v - (-10.0)).abs() < 1e-3));
    }

    #[test]
    fn repack_awq_gptq_rejects_bad_dimensions() {
        // K not divisible by 8.
        let input = AwqGptqInput {
            qweight: vec![0u32; 2],
            qzeros: vec![],
            scales: vec![f16::from_f32(1.0); 2],
            out_features: 2,
            in_features: 12, // not % 8
            group_size: 4,
            scheme: QuantScheme::AWQ,
        };
        assert!(repack_awq_gptq(&input).is_err());

        // Wrong qweight length.
        let input = AwqGptqInput {
            qweight: vec![0u32; 3], // expected (32/8)*2 = 8
            qzeros: vec![],
            scales: vec![f16::from_f32(1.0); 2 * 8],
            out_features: 2,
            in_features: 32,
            group_size: 4,
            scheme: QuantScheme::AWQ,
        };
        assert!(repack_awq_gptq(&input).is_err());

        // Wrong scales length.
        let input = AwqGptqInput {
            qweight: vec![0u32; 8],
            qzeros: vec![],
            scales: vec![f16::from_f32(1.0); 10], // expected 16
            out_features: 2,
            in_features: 32,
            group_size: 4,
            scheme: QuantScheme::AWQ,
        };
        assert!(repack_awq_gptq(&input).is_err());
    }
}
