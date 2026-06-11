/// Borrowed CUDA stream handle. `0` is the CUDA default stream.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StreamHandle(u64);

impl StreamHandle {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn null() -> Self {
        Self(0)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[cfg(feature = "cuda")]
    pub fn as_ffi(self) -> ayaka_kernel_api::ffi::AyakaStream {
        self.0 as usize as ayaka_kernel_api::ffi::AyakaStream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_stream_is_raw_zero() {
        let stream = StreamHandle::null();
        assert!(stream.is_null());
        assert_eq!(stream.raw(), 0);
    }
}
