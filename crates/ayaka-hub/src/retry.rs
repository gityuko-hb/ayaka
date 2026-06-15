//! Retry policy and helper for transient network failures.
//!
//! Used by [`HubClient`](crate::HubClient) to retry operations that fail with
//! connection-reset, timeout, or 5xx / 429 errors. Resolution order for the
//! retry parameters:
//!
//! 1. `HubClientBuilder::with_max_retries(n)` / `with_retry_base_delay_ms(ms)`
//!    if explicitly set on the builder (highest priority).
//! 2. `AYAKA_HUB_MAX_RETRIES` / `AYAKA_HUB_RETRY_BASE_DELAY_MS` env vars.
//! 3. Defaults: `max_retries = 0`, `base_delay_ms = 100`.

use std::time::Duration;

use crate::error::HubError;

/// Retry parameters applied to every blocking operation in [`HubClient`](crate::HubClient).
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts after the first failure. `0` disables retry.
    pub max_retries: u32,
    /// Base delay for exponential backoff. Delay after attempt `n` is
    /// `base_delay_ms * 2^n`, capped at 5 seconds.
    pub base_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            base_delay_ms: 100,
        }
    }
}

const MAX_BACKOFF_MS: u64 = 5_000;

/// Return `true` when an error is transient and worth retrying.
pub fn is_retryable(err: &HubError) -> bool {
    match err {
        HubError::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::Interrupted
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::BrokenPipe
        ),
        HubError::Http { code, .. } => matches!(*code, 429 | 500..=599),
        HubError::Api(msg) => retryable_message(msg),
        HubError::Auth(_) | HubError::NotFound(_) | HubError::Offline(_) | HubError::Cache(_) => {
            false
        },
    }
}

/// Substring matcher for the opaque `HubError::Api(String)` variant. The HF
/// client does not always surface structured I/O errors, so we fall back to
/// matching known transient message fragments.
fn retryable_message(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    const NEEDLES: &[&str] = &[
        "connection reset",
        "connection forcibly closed",
        "connection aborted",
        "connection refused",
        "broken pipe",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "try again",
        "os error 10053", // WSAECONNABORTED (Windows)
        "os error 10054", // WSAECONNRESET  (Windows)
        "os error 10060", // WSAETIMEDOUT   (Windows)
        "os error 10061", // WSAECONNREFUSED (Windows)
        "io:",
    ];
    NEEDLES.iter().any(|n| lower.contains(n))
}

/// Compute the backoff duration for retry attempt `n` (zero-indexed).
#[inline]
pub fn backoff_for(
    policy: RetryPolicy,
    attempt: u32,
) -> Duration {
    let delay = policy
        .base_delay_ms
        .saturating_mul(1u64 << attempt.min(20));
    Duration::from_millis(delay.min(MAX_BACKOFF_MS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_io_connection_reset() {
        let e = HubError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset",
        ));
        assert!(is_retryable(&e));
    }

    #[test]
    fn retryable_io_timed_out() {
        let e = HubError::Io(std::io::Error::new(std::io::ErrorKind::TimedOut, "to"));
        assert!(is_retryable(&e));
    }

    #[test]
    fn retryable_http_5xx() {
        let e = HubError::Http {
            code: 503,
            message: "unavailable".into(),
        };
        assert!(is_retryable(&e));
    }

    #[test]
    fn retryable_http_429() {
        let e = HubError::Http {
            code: 429,
            message: "rate limited".into(),
        };
        assert!(is_retryable(&e));
    }

    #[test]
    fn not_retryable_http_4xx() {
        let e = HubError::Http {
            code: 404,
            message: "not found".into(),
        };
        assert!(!is_retryable(&e));
    }

    #[test]
    fn not_retryable_auth() {
        assert!(!is_retryable(&HubError::Auth(401)));
    }

    #[test]
    fn not_retryable_not_found() {
        assert!(!is_retryable(&HubError::NotFound("x".into())));
    }

    #[test]
    fn not_retryable_offline() {
        assert!(!is_retryable(&HubError::Offline("x".into())));
    }

    #[test]
    fn not_retryable_cache() {
        assert!(!is_retryable(&HubError::Cache("x".into())));
    }

    #[test]
    fn retryable_api_connection_reset_msg() {
        let e = HubError::Api("request error: io: Connection reset by peer".into());
        assert!(is_retryable(&e));
    }

    #[test]
    fn retryable_api_wsa_error_msg() {
        let e = HubError::Api(
            "request error: io: An existing connection was forcibly closed by the remote host. (os error 10054)".into(),
        );
        assert!(is_retryable(&e));
    }

    #[test]
    fn retryable_api_timed_out_msg() {
        let e = HubError::Api("operation timed out".into());
        assert!(is_retryable(&e));
    }

    #[test]
    fn not_retryable_generic_api_msg() {
        let e = HubError::Api("not found: missing.json".into());
        assert!(!is_retryable(&e));
    }

    #[test]
    fn backoff_grows_exponentially() {
        let p = RetryPolicy {
            max_retries: 5,
            base_delay_ms: 100,
        };
        assert_eq!(backoff_for(p, 0), Duration::from_millis(100));
        assert_eq!(backoff_for(p, 1), Duration::from_millis(200));
        assert_eq!(backoff_for(p, 2), Duration::from_millis(400));
        assert_eq!(backoff_for(p, 3), Duration::from_millis(800));
    }

    #[test]
    fn backoff_capped_at_5s() {
        let p = RetryPolicy {
            max_retries: 20,
            base_delay_ms: 100,
        };
        assert_eq!(backoff_for(p, 10), Duration::from_millis(5_000));
        assert_eq!(backoff_for(p, 20), Duration::from_millis(5_000));
    }

    #[test]
    fn default_policy_disables_retry() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 0);
        assert_eq!(p.base_delay_ms, 100);
    }
}
