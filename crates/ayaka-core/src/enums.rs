//! Shared op-configuration enums used by the descriptors and mirrored in the C
//! headers (`include/ayaka/descriptors/*`). All `#[repr(u8)]` with explicit
//! discriminants — keep in sync across the FFI boundary.

/// Attention masking mode.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum MaskMode {
    /// No mask (bidirectional / encoder).
    Full = 0,
    /// Causal lower-triangular mask.
    Causal = 1,
    /// Causal + left window of `window_left` tokens.
    SlidingWindow = 2,
    /// Caller-provided custom mask (e.g. speculative tree mask).
    Custom = 3,
}

/// Positional-encoding variant applied inside / before attention.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum PosEncoding {
    None = 0,
    /// RoPE, GPT-NeoX rotation layout.
    RopeNeox = 1,
    /// RoPE, GPT-J (interleaved) layout.
    RopeGptj = 2,
    /// Llama-3 rope scaling.
    Llama3 = 3,
    /// YaRN context extension.
    Yarn = 4,
    /// LongRoPE.
    LongRope = 5,
    /// NTK-aware scaling.
    Ntk = 6,
    /// ALiBi linear biases.
    Alibi = 7,
}

/// Per-request sampling method.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum SamplingMethod {
    Greedy = 0,
    Random = 1,
}

/// Which attention kernel backend to dispatch to. `Auto` lets the native
/// dispatch table choose based on (dtype, head_dim, SM arch, mask, page_size).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum AttentionBackend {
    Auto = 0,
    /// Hand-written Ayaka kernels.
    Native = 1,
    /// Wrapped FlashInfer cubins.
    FlashInfer = 2,
    /// Wrapped DeepSeek FlashMLA (MLA decode).
    FlashMla = 3,
    /// CUTLASS MLA backend.
    CutlassMla = 4,
}
