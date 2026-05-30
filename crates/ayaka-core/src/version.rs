pub const HELIX_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const HELIX_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

impl Version {
    pub const fn new(
        major: u8,
        minor: u8,
        patch: u8,
    ) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

pub const CURRENT_VERSION: Version = Version::new(0, 1, 0);
