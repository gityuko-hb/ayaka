//! Devices and CUDA compute capabilities.
//!
//! [`SmArch`] is the input to the multi-SM kernel dispatch table; its helpers
//! (`supports_tma`, `supports_wgmma`, `supports_tcgen05`, ...) gate which kernel
//! variant the launcher selects. Mirrors `include/ayaka/device.h`.

/// Device backend kind.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum DeviceKind {
    Cpu = 0,
    Cuda = 1,
}

/// A concrete device = (kind, ordinal).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct Device {
    pub kind: DeviceKind,
    pub ordinal: u16,
}

impl Device {
    pub const CPU: Self = Self {
        kind: DeviceKind::Cpu,
        ordinal: 0,
    };

    #[inline]
    pub const fn cuda(ordinal: u16) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            ordinal,
        }
    }

    #[inline]
    pub const fn is_cuda(self) -> bool {
        matches!(self.kind, DeviceKind::Cuda)
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

/// CUDA compute capability, e.g. `SmArch { major: 9, minor: 0 }` for Hopper.
#[derive(
    Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct SmArch {
    pub major: u8,
    pub minor: u8,
}

impl SmArch {
    pub const SM_80: Self = Self { major: 8, minor: 0 }; // A100
    pub const SM_86: Self = Self { major: 8, minor: 6 }; // A10/RTX 30xx
    pub const SM_89: Self = Self { major: 8, minor: 9 }; // Ada / RTX 40xx
    pub const SM_90: Self = Self { major: 9, minor: 0 }; // Hopper H100/H200
    pub const SM_100: Self = Self {
        major: 10,
        minor: 0,
    }; // Blackwell B100/B200
    pub const SM_103: Self = Self {
        major: 10,
        minor: 3,
    }; // Blackwell GB200-class
    pub const SM_120: Self = Self {
        major: 12,
        minor: 0,
    }; // Blackwell consumer / RTX 50xx
    pub const SM_121: Self = Self {
        major: 12,
        minor: 1,
    }; // DGX Spark-class

    #[inline]
    pub const fn new(
        major: u8,
        minor: u8,
    ) -> Self {
        Self { major, minor }
    }

    /// Packed capability number, e.g. 9.0 -> 90, 10.3 -> 103.
    #[inline]
    pub const fn capability(self) -> u32 {
        self.major as u32 * 10 + self.minor as u32
    }

    /// FP8 tensor-core support starts at Ada (sm_89).
    #[inline]
    pub const fn supports_fp8(self) -> bool {
        self.capability() >= 89
    }

    /// FP4 / 5th-gen tensor cores: Blackwell and later.
    #[inline]
    pub const fn supports_fp4(self) -> bool {
        self.major >= 10
    }

    /// Tensor Memory Accelerator (bulk async copy): Hopper and later.
    #[inline]
    pub const fn supports_tma(self) -> bool {
        self.capability() >= 90
    }

    /// Warpgroup MMA (`wgmma`): Hopper only. Deprecated on Blackwell in favor of tcgen05.
    #[inline]
    pub const fn supports_wgmma(self) -> bool {
        self.major == 9
    }

    /// 5th-gen tensor-core MMA (`tcgen05` / UMMA + TMEM): Blackwell and later.
    #[inline]
    pub const fn supports_tcgen05(self) -> bool {
        self.major >= 10
    }

    /// Async copy (`cp.async`): Ampere and later.
    #[inline]
    pub const fn supports_cp_async(self) -> bool {
        self.capability() >= 80
    }

    pub const fn family(self) -> SmFamily {
        match (self.major, self.minor) {
            (8, 9) => SmFamily::Ada,
            (8, _) => SmFamily::Ampere,
            (9, _) => SmFamily::Hopper,
            (10..=12, _) => SmFamily::Blackwell,
            _ => SmFamily::Unknown,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum SmFamily {
    Ampere = 0,
    Ada = 1,
    Hopper = 2,
    Blackwell = 3,
    Unknown = 255,
}
