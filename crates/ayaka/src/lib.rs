//! ```
//! use ayaka::{AyakaError, TokenEvent};
//!
//! fn assert_public_event_type(_: Option<TokenEvent>) {}
//! fn assert_public_error_type(_: Option<AyakaError>) {}
//!
//! assert_public_event_type(None);
//! assert_public_error_type(None);
//! ```

use ayaka_core::id::TokenId;
use ayaka_executor::{MockExecution, MockExecutor};
use ayaka_scheduler::SimpleScheduler;
use ayaka_telemetry::TelemetryCounters;

pub use ayaka_error::AyakaError;
pub use ayaka_executor::MockExecutorConfig;
pub use ayaka_request::{
    FinishReason, RequestError, RequestState, RequestStatus, SequenceState, TokenEvent,
};
pub use ayaka_scheduler::SchedulerConfig;
pub use ayaka_server::GenerateRequest;
pub use ayaka_telemetry::TelemetrySnapshot;

#[derive(Debug, Clone)]
pub struct MockEngineConfig {
    pub scheduler: SchedulerConfig,
    pub executor: MockExecutorConfig,
}

impl MockEngineConfig {
    pub fn new(
        max_batch_size: usize,
        mock_token_ids: Vec<TokenId>,
    ) -> Result<Self, AyakaError> {
        let scheduler = SchedulerConfig::new(max_batch_size)
            .map_err(|err| AyakaError::invalid_argument(err.to_string()))?;
        Ok(Self {
            scheduler,
            executor: MockExecutorConfig::new(mock_token_ids),
        })
    }
}

#[derive(Debug)]
pub struct AyakaEngine {
    scheduler: SimpleScheduler,
    executor: MockExecutor,
    telemetry: TelemetryCounters,
}

impl AyakaEngine {
    pub fn new_mock(config: MockEngineConfig) -> Self {
        Self {
            scheduler: SimpleScheduler::new(config.scheduler),
            executor: MockExecutor::new(config.executor),
            telemetry: TelemetryCounters::default(),
        }
    }

    pub fn generate_stream(
        &mut self,
        request: GenerateRequest,
    ) -> Result<Vec<TokenEvent>, AyakaError> {
        let state = request.into_request_state().map_err(|err| {
            self.telemetry.record_rejected_request();
            AyakaError::invalid_argument(err.to_string())
        })?;

        self.scheduler.admit(state).map_err(|err| {
            self.telemetry.record_rejected_request();
            AyakaError::invalid_argument(err.to_string())
        })?;
        self.telemetry.record_admitted_request();

        let Some(batch) = self.scheduler.next_batch() else {
            return Ok(Vec::new());
        };

        self.telemetry.record_created_batch();
        let execution = self
            .executor
            .execute(batch.into_requests())
            .map_err(|err| {
                AyakaError::internal(format!(
                    "executor failed in ayaka::AyakaEngine::generate_stream: {}",
                    err
                ))
            })?;
        self.record_execution(&execution);
        Ok(execution.events)
    }

    pub fn telemetry(&self) -> TelemetrySnapshot {
        self.telemetry.snapshot()
    }

    fn record_execution(
        &mut self,
        execution: &MockExecution,
    ) {
        for event in &execution.events {
            if event.token_id.is_some() {
                self.telemetry.record_emitted_token();
            }
            if event.finish_reason.is_some() {
                self.telemetry.record_completed_request();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_engine_streams_tokens_and_records_telemetry() {
        let config = MockEngineConfig::new(2, vec![10, 11, 12]).unwrap();
        let mut engine = AyakaEngine::new_mock(config);

        let events = engine
            .generate_stream(GenerateRequest::new("hello", vec![1, 2], 2, true))
            .unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].token_id, Some(10));
        assert_eq!(events[1].token_id, Some(11));
        assert_eq!(events[2].finish_reason, Some(FinishReason::Length));

        let telemetry = engine.telemetry();
        assert_eq!(telemetry.admitted_requests, 1);
        assert_eq!(telemetry.rejected_requests, 0);
        assert_eq!(telemetry.created_batches, 1);
        assert_eq!(telemetry.emitted_tokens, 2);
        assert_eq!(telemetry.completed_requests, 1);
    }

    #[test]
    fn invalid_generate_request_records_rejection() {
        let config = MockEngineConfig::new(1, vec![10]).unwrap();
        let mut engine = AyakaEngine::new_mock(config);

        let err = engine
            .generate_stream(GenerateRequest::new("hello", Vec::new(), 2, true))
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("prompt_tokens must not be empty")
        );
        let telemetry = engine.telemetry();
        assert_eq!(telemetry.admitted_requests, 0);
        assert_eq!(telemetry.rejected_requests, 1);
    }

    #[test]
    fn zero_budget_generate_request_records_rejection() {
        let config = MockEngineConfig::new(1, vec![10]).unwrap();
        let mut engine = AyakaEngine::new_mock(config);

        let err = engine
            .generate_stream(GenerateRequest::new("hello", vec![1], 0, true))
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("max_new_tokens must be greater than zero")
        );
        let telemetry = engine.telemetry();
        assert_eq!(telemetry.admitted_requests, 0);
        assert_eq!(telemetry.rejected_requests, 1);
    }

    #[test]
    fn facade_reexports_public_signature_types() {
        fn assert_public_event_type(_: Option<TokenEvent>) {}
        fn assert_public_error_type(_: Option<AyakaError>) {}
        fn assert_public_scheduler_config(_: Option<SchedulerConfig>) {}
        fn assert_public_executor_config(_: Option<MockExecutorConfig>) {}
        fn assert_public_telemetry_snapshot(_: Option<TelemetrySnapshot>) {}

        assert_public_event_type(None);
        assert_public_error_type(None);
        assert_public_scheduler_config(None);
        assert_public_executor_config(None);
        assert_public_telemetry_snapshot(None);
    }

    #[test]
    fn mock_engine_config_rejects_zero_batch_size() {
        let err = MockEngineConfig::new(0, vec![10]).unwrap_err();
        assert!(
            err.to_string()
                .contains("max_batch_size must be greater than zero")
        );
    }
}
