#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub admitted_requests: usize,
    pub rejected_requests: usize,
    pub created_batches: usize,
    pub emitted_tokens: usize,
    pub completed_requests: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TelemetryCounters {
    snapshot: TelemetrySnapshot,
}

impl TelemetryCounters {
    pub fn record_admitted_request(&mut self) {
        self.snapshot.admitted_requests += 1;
    }

    pub fn record_rejected_request(&mut self) {
        self.snapshot.rejected_requests += 1;
    }

    pub fn record_created_batch(&mut self) {
        self.snapshot.created_batches += 1;
    }

    pub fn record_emitted_token(&mut self) {
        self.snapshot.emitted_tokens += 1;
    }

    pub fn record_completed_request(&mut self) {
        self.snapshot.completed_requests += 1;
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        self.snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_record_request_lifecycle_events() {
        let mut counters = TelemetryCounters::default();

        counters.record_admitted_request();
        counters.record_rejected_request();
        counters.record_created_batch();
        counters.record_emitted_token();
        counters.record_emitted_token();
        counters.record_completed_request();

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.admitted_requests, 1);
        assert_eq!(snapshot.rejected_requests, 1);
        assert_eq!(snapshot.created_batches, 1);
        assert_eq!(snapshot.emitted_tokens, 2);
        assert_eq!(snapshot.completed_requests, 1);
    }
}
