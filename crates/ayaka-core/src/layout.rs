//! Memory layout: strides, contiguity, and the KV-cache physical layout.

use crate::shape::{MAX_RANK, Shape};

/// High-level memory format hint.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemoryFormat {
    RowMajor = 0,
    ColMajor = 1,
    Strided = 2,
}

/// Physical layout of the paged KV cache.
/// NHD is the natural FlashInfer default;
/// HND coalesces better for FP8 KV. Mirrors `ayaka_kv_layout_t`.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum KvLayout {
    /// `(.., seq, head, dim)`
    Nhd = 0,
    /// `(.., head, seq, dim)`
    Hnd = 1,
}

/// Element strides for each dimension (alloc-free, fixed capacity).
#[derive(Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Strides {
    strides: [i64; MAX_RANK],
    rank: u8,
}

impl Strides {
    /// Row-major (C-contiguous) strides for `shape`.
    pub fn contiguous(shape: &Shape) -> Self {
        let r = shape.rank();
        let mut s = [0i64; MAX_RANK];
        let mut acc = 1i64;
        let mut i = r;
        while i > 0 {
            i -= 1;
            s[i] = acc;
            acc *= shape.dim(i);
        }
        Self {
            strides: s,
            rank: r as u8,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[i64] {
        &self.strides[..self.rank as usize]
    }

    pub fn is_contiguous(
        &self,
        shape: &Shape,
    ) -> bool {
        self.rank as usize == shape.rank() && *self == Self::contiguous(shape)
    }
}

impl core::fmt::Debug for Strides {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "Strides{:?}", self.as_slice())
    }
}
