//! Convenience macros for early error returns.

/// Returns an `Err(AyakaError)` with the given kind and formatted message.
#[macro_export]
macro_rules! ayaka_bail {
    ($kind:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        return ::core::result::Result::Err(
            $crate::AyakaError::new($kind, ::alloc::format!($fmt $(, $arg)*))
        )
    };
}

/// Ensures a condition is true, otherwise returns an `Err(AyakaError)`.
#[macro_export]
macro_rules! ayaka_ensure {
    ($cond:expr, $kind:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        if !$cond {
            $crate::ayaka_bail!($kind, $fmt $(, $arg)*);
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{ErrorKind, Result};

    #[test]
    fn ayaka_ensure_passes() {
        fn check() -> Result<()> {
            crate::ayaka_ensure!(1 + 1 == 2, ErrorKind::Internal, "math broke");
            Ok(())
        }
        assert!(check().is_ok());
    }

    #[test]
    fn ayaka_ensure_fails() {
        fn check() -> Result<()> {
            crate::ayaka_ensure!(1 + 1 == 3, ErrorKind::Internal, "math broke: {}", 42);
            Ok(())
        }
        let err = check().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Internal);
        assert!(err.message().contains("math broke: 42"));
    }
}
