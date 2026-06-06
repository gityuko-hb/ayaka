//! Monotonic clock and per-request timing for the benchmark axes (TTFT, ITL/TPOT,
//! end-to-end). No `Instant` in serializable types so timings can be logged.

use std::time::Instant;

/// Nanoseconds since a [`Clock`] epoch.
#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Timestamp(pub u64);

impl Timestamp {
    #[inline]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
    #[inline]
    pub fn as_micros_f64(self) -> f64 {
        self.0 as f64 / 1_000.0
    }
    #[inline]
    pub fn as_millis_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

/// Monotonic clock anchored at construction. Cheap `now()` via `Instant::elapsed`.
pub struct Clock {
    start: Instant,
}

impl Clock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    #[inline]
    pub fn now(&self) -> Timestamp {
        Timestamp(self.start.elapsed().as_nanos() as u64)
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing checkpoints for one request, in clock nanoseconds.
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestTiming {
    pub arrival_ns: u64,
    pub first_scheduled_ns: u64,
    pub first_token_ns: u64,
    pub last_token_ns: u64,
    pub num_output_tokens: u32,
}

impl RequestTiming {
    /// Time to first token (arrival -> first generated token).
    pub fn ttft_ns(&self) -> Option<u64> {
        (self.first_token_ns >= self.arrival_ns && self.first_token_ns != 0)
            .then(|| self.first_token_ns - self.arrival_ns)
    }

    /// Mean time per output token (a.k.a. inter-token latency), excluding the
    /// first token. Needs at least 2 output tokens.
    pub fn tpot_ns(&self) -> Option<u64> {
        if self.num_output_tokens >= 2 && self.last_token_ns > self.first_token_ns {
            Some((self.last_token_ns - self.first_token_ns) / (self.num_output_tokens as u64 - 1))
        } else {
            None
        }
    }

    /// End-to-end latency (arrival -> last token).
    pub fn e2e_ns(&self) -> Option<u64> {
        (self.last_token_ns >= self.arrival_ns && self.last_token_ns != 0)
            .then(|| self.last_token_ns - self.arrival_ns)
    }
}
