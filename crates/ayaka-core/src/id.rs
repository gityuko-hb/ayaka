use core::fmt;
use std::sync::atomic;

macro_rules! define_new_id_type {
    ($name:ident, $inner:ty, $invalid:expr, $prefix:expr, $atomic_ty:ty) => {
        #[repr(transparent)]
        #[derive(
            Copy,
            Clone,
            PartialEq,
            Eq,
            Hash,
            PartialOrd,
            Ord,
            Default,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(pub $inner);

        impl $name {
            pub const INVALID: Self = Self($invalid);
            pub const ZERO: Self = Self(0);

            #[inline]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            pub fn generate() -> Self {
                static COUNTER: $atomic_ty = <$atomic_ty>::new(1);
                Self(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            }

            #[inline]
            pub const fn raw(self) -> $inner {
                self.0
            }

            #[inline]
            pub const fn is_valid(self) -> bool {
                self.0 != $invalid
            }

            #[inline]
            pub const fn is_invalid(self) -> bool {
                self.0 == $invalid
            }
        }

        impl fmt::Display for $name {
            fn fmt(
                &self,
                f: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                if self.is_invalid() {
                    write!(f, "{}-INVALID", $prefix)
                } else {
                    write!(f, "{}-{}", $prefix, self.0)
                }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(
                &self,
                f: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                if self.is_invalid() {
                    write!(f, "{}(INVALID)", stringify!($name))
                } else {
                    write!(f, "{}({})", stringify!($name), self.0)
                }
            }
        }
    };
}

define_new_id_type!(RequestId, u64, u64::MAX, "req", atomic::AtomicU64);
define_new_id_type!(SequenceId, u64, u64::MAX, "seq", atomic::AtomicU64);
define_new_id_type!(BatchId, u64, u64::MAX, "batch", atomic::AtomicU64);

define_new_id_type!(NodeId, u32, u32::MAX, "node", atomic::AtomicU32);
define_new_id_type!(SpecNodeId, u32, u32::MAX, "spec-node", atomic::AtomicU32);
define_new_id_type!(PageId, u32, u32::MAX, "page", atomic::AtomicU32);
define_new_id_type!(BlockId, u32, u32::MAX, "block", atomic::AtomicU32);

define_new_id_type!(LayerId, u16, u16::MAX, "layer", atomic::AtomicU16);
define_new_id_type!(HeadId, u16, u16::MAX, "head", atomic::AtomicU16);
define_new_id_type!(DeviceId, u16, u16::MAX, "device", atomic::AtomicU16);
define_new_id_type!(StreamId, u16, u16::MAX, "stream", atomic::AtomicU16);
define_new_id_type!(GraphId, u32, u32::MAX, "graph", atomic::AtomicU32);

pub type TokenId = u32;
pub type Position = u32;
pub type Epoch = u64;
pub type Generation = u64;
