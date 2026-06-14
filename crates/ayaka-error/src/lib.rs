//! Shared error handling for the Ayaka inference engine.
//!
//! This crate is intentionally lightweight and `no_std`-compatible when the
//! `std` feature is disabled. It provides stable FFI status codes, a domain
//! error type, context helpers, and conversion to/from the C ABI boundary.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod ext;
pub mod ffi;
pub mod macros;
pub mod status;

pub use error::{AyakaError, ErrorKind, Result};
pub use ext::{OptionExt, ResultExt};
pub use ffi::AyakaStatus;
pub use status::StatusCode;
