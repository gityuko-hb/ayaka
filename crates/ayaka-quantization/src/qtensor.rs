//! In-memory packed quantized tensor.
//!
//! [`QTensor`] holds the raw packed block bytes (quantized weights + scales)
//! without dequantizing. This is the Rust-side representation that will be
//! uploaded to the GPU as a `DType::U8` tensor so CUDA kernels can dequantize
//! on-the-fly during GEMM (Phase 1.1 goal).

use crate::gguf_block::{GgufBlock, GgufDequantError};
use crate::scheme::QuantScheme;

/// In-memory packed quantized tensor: raw block bytes + shape + scheme metadata.
///
/// The `bytes` field holds the exact on-disk GGUF block layout (e.g., Q4_K
/// super-blocks of 144 bytes each). The `dims` are in Hugging Face order
/// (`[out, in, ...]`), matching `GgufTensorInfo::dims`.
#[derive(Debug, Clone)]
pub struct QTensor {
    /// Raw packed block bytes (quants + scales, exact GGUF on-disk layout).
    pub bytes: Vec<u8>,
    /// Dimensions in Hugging Face order (`[out, in, ...]`).
    pub dims: Vec<usize>,
    /// Quantization scheme (Q4K, Q6K, Q4_0, Q8_0, etc.).
    pub scheme: QuantScheme,
    /// Block geometry: weights per block and bytes per block.
    pub block: GgufBlock,
}

impl QTensor {
    /// Build a [`QTensor`] from raw block bytes, shape, and scheme.
    ///
    /// Validates that `bytes.len()` exactly matches the block-packed size
    /// implied by `dims`: `n_blocks * bytes_per_block` where
    /// `n_blocks = num_elements.div_ceil(weights_per_block)`.
    pub fn from_raw(
        bytes: Vec<u8>,
        dims: Vec<usize>,
        scheme: QuantScheme,
    ) -> Result<Self, GgufDequantError> {
        let dtype = scheme.to_gguf_dtype();
        let block = dtype.block();
        let n_elements: usize = dims.iter().product();
        let n_blocks = n_elements.div_ceil(block.weights_per_block);
        let expected = n_blocks * block.bytes_per_block;
        if bytes.len() != expected {
            return Err(GgufDequantError::MalformedLength {
                dtype,
                raw_len: bytes.len(),
                expected,
            });
        }
        Ok(Self {
            bytes,
            dims,
            scheme,
            block,
        })
    }

    /// Total scalar element count (product of `dims`).
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// On-disk byte size of the stored blocks (same as `bytes.len()`).
    #[inline]
    pub fn stored_bytes(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qtensor_from_raw_bytes_preserves_geometry() {
        let q = QTensor::from_raw(vec![0u8; 144], vec![1, 256], QuantScheme::Q4K)
            .expect("Q4K single super-block should construct");
        assert_eq!(q.scheme, QuantScheme::Q4K);
        assert_eq!(
            q.block,
            GgufBlock {
                weights_per_block: 256,
                bytes_per_block: 144,
            }
        );
        assert_eq!(q.dims, vec![1, 256]);
        assert_eq!(q.bytes.len(), 144);
        assert_eq!(q.num_elements(), 256);
    }

    #[test]
    fn qtensor_byte_size_matches_block_count() {
        // Q6K: dims [8, 512] = 4096 elements -> 4096/256 = 16 super-blocks
        // -> 16 * 210 = 3360 bytes.
        let q = QTensor::from_raw(vec![0u8; 3360], vec![8, 512], QuantScheme::Q6K)
            .expect("Q6K 16 super-blocks should construct");
        assert_eq!(q.stored_bytes(), 3360);
        assert_eq!(q.num_elements(), 4096);
    }

    #[test]
    fn qtensor_rejects_short_buffer() {
        // dims [1, 256] = 256 elements -> 1 Q4K super-block -> 144 bytes needed;
        // 100 bytes is short of the exact requirement, so reject.
        let err = QTensor::from_raw(vec![0u8; 100], vec![1, 256], QuantScheme::Q4K).unwrap_err();
        assert!(matches!(err, GgufDequantError::MalformedLength { .. }));
    }

    #[test]
    fn qtensor_q4_0_works() {
        // Q4_0: dims [2, 32] = 64 elements -> 2 blocks * 18 bytes = 36 bytes.
        let q = QTensor::from_raw(vec![0u8; 36], vec![2, 32], QuantScheme::Q4_0)
            .expect("Q4_0 two blocks should construct");
        assert_eq!(q.block.weights_per_block, 32);
        assert_eq!(q.block.bytes_per_block, 18);
        assert_eq!(q.stored_bytes(), 36);
    }
}
