bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    pub struct TensorFlags: u32 {
        const NONE        = 0;
        const CONTIGUOUS  = 1 << 0;
        const READ_ONLY   = 1 << 1;
        const OWNS_DATA   = 1 << 2;
        const ALIASED     = 1 << 3;
        const PINNED      = 1 << 4;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    pub struct RequestFlags: u32 {
        const NONE              = 0;
        const STREAMING         = 1 << 0;
        const USE_PREFIX_CACHE  = 1 << 1;
        const USE_SPECULATIVE   = 1 << 2;
        const HIGH_PRIORITY     = 1 << 3;
        const CANCELLED         = 1 << 4;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    pub struct KernelFlags: u32 {
        const NONE             = 0;
        const USE_CUDA_GRAPH   = 1 << 0;
        const USE_TENSOR_CORES = 1 << 1;
        const USE_FP8          = 1 << 2;
        const USE_QUANTIZED_KV = 1 << 3;
    }
}
