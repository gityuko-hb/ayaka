//! Radix prefix caching for cross-request KV reuse.

pub mod radix;

pub use radix::{PrefixMatch, RadixCache};
