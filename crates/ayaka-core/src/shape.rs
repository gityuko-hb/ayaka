//! Fixed-capacity tensor shape (alloc-free; lives on the stack and crosses FFI).

use core::fmt;

/// Maximum tensor rank. Inference tensors are rank <= 5 in practice; 8 leaves
/// headroom and keeps [`Shape`] / `ayaka_tensor_view_t` a fixed size.
pub const MAX_RANK: usize = 8;

/// An inline, copyable tensor shape. Dims are `i64` to match the C ABI and to
/// avoid `usize`/`i32` conversion bugs at the boundary.
#[derive(Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Shape {
    dims: [i64; MAX_RANK],
    rank: u8,
}

impl Shape {
    pub const fn scalar() -> Self {
        Self {
            dims: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// Build from a slice of dims. Panics if `dims.len() > MAX_RANK`.
    pub fn new(dims: &[i64]) -> Self {
        assert!(
            dims.len() <= MAX_RANK,
            "rank {} exceeds MAX_RANK",
            dims.len()
        );
        let mut d = [0i64; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Self {
            dims: d,
            rank: dims.len() as u8,
        }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    #[inline]
    pub fn dims(&self) -> &[i64] {
        &self.dims[..self.rank as usize]
    }

    #[inline]
    pub fn dim(
        &self,
        i: usize,
    ) -> i64 {
        self.dims[i]
    }

    /// Total number of elements (rank-0 scalar -> 1).
    pub fn numel(&self) -> i64 {
        self.dims().iter().product()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.rank == 0
    }
}

impl fmt::Debug for Shape {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "Shape{:?}", self.dims())
    }
}
