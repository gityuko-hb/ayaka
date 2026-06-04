use ayaka_core::id::TokenId;
use ayaka_request::{FinishReason, RequestResult, RequestState, RequestStatus, TokenEvent};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockExecutorConfig {
    pub token_ids: Vec<TokenId>,
}

impl MockExecutorConfig {
    pub fn new(token_ids: Vec<TokenId>) -> Self {
        Self { token_ids }
    }
}

#[derive(Debug, Clone)]
pub struct MockExecutor {
    config: MockExecutorConfig,
}

impl MockExecutor {
    pub fn new(config: MockExecutorConfig) -> Self {
        Self { config }
    }

    pub fn execute(
        &self,
        requests: Vec<RequestState>,
    ) -> RequestResult<MockExecution> {
        let mut completed_requests = Vec::with_capacity(requests.len());
        let mut events = Vec::new();

        for mut request in requests {
            if request.status == RequestStatus::Queued {
                request.mark_running()?;
            }

            let budget = request.remaining_new_tokens();
            let output_len = budget.min(self.config.token_ids.len());

            for (index, token_id) in self
                .config
                .token_ids
                .iter()
                .take(output_len)
                .enumerate()
            {
                request.record_generated_token(*token_id)?;
                events.push(TokenEvent::token(
                    request.id,
                    request.sequence.id,
                    *token_id,
                    Some(format!("token:{token_id}")),
                    index,
                ));
            }

            let finish_reason = if output_len == budget {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            request.mark_completed(finish_reason)?;
            events.push(TokenEvent::finished(
                request.id,
                request.sequence.id,
                output_len,
                finish_reason,
            ));

            completed_requests.push(request);
        }

        Ok(MockExecution {
            requests: completed_requests,
            events,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockExecution {
    pub requests: Vec<RequestState>,
    pub events: Vec<TokenEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_core::id::{RequestId, SequenceId};
    use ayaka_request::{FinishReason, RequestState, RequestStatus};

    fn running_request(max_new_tokens: usize) -> RequestState {
        let mut request = RequestState::from_parts(
            RequestId::new(9),
            SequenceId::new(10),
            "hello".into(),
            vec![1],
            max_new_tokens,
            true,
        )
        .unwrap();
        request.mark_queued().unwrap();
        request.mark_running().unwrap();
        request
    }

    #[test]
    fn mock_executor_emits_token_events_and_length_finish() {
        let executor = MockExecutor::new(MockExecutorConfig::new(vec![7, 8, 9]));
        let execution = executor
            .execute(vec![running_request(2)])
            .unwrap();

        assert_eq!(execution.events.len(), 3);
        assert_eq!(execution.events[0].token_id, Some(7));
        assert_eq!(execution.events[0].text_delta.as_deref(), Some("token:7"));
        assert_eq!(execution.events[1].token_id, Some(8));
        assert_eq!(
            execution.events[2].finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(execution.requests[0].generated_tokens, vec![7, 8]);
        assert_eq!(execution.requests[0].status, RequestStatus::Completed);
    }

    #[test]
    fn mock_executor_uses_stop_when_mock_tokens_end_before_budget() {
        let executor = MockExecutor::new(MockExecutorConfig::new(vec![7]));
        let execution = executor
            .execute(vec![running_request(3)])
            .unwrap();

        assert_eq!(execution.events.len(), 2);
        assert_eq!(execution.events[0].token_id, Some(7));
        assert_eq!(execution.events[1].finish_reason, Some(FinishReason::Stop));
        assert_eq!(execution.requests[0].remaining_new_tokens(), 0);
    }

    #[test]
    fn mock_executor_accepts_queued_request_and_completes_it() {
        let mut request = RequestState::from_parts(
            RequestId::new(11),
            SequenceId::new(12),
            "hello".into(),
            vec![1],
            1,
            true,
        )
        .unwrap();
        request.mark_queued().unwrap();
        let executor = MockExecutor::new(MockExecutorConfig::new(vec![5]));

        let execution = executor.execute(vec![request]).unwrap();

        assert_eq!(execution.requests[0].status, RequestStatus::Completed);
        assert_eq!(execution.requests[0].generated_tokens, vec![5]);
        assert_eq!(
            execution.requests[0].finish_reason,
            Some(FinishReason::Length)
        );
    }

    #[test]
    fn mock_executor_empty_tokens_finishes_with_stop() {
        let executor = MockExecutor::new(MockExecutorConfig::new(Vec::new()));

        let execution = executor
            .execute(vec![running_request(2)])
            .unwrap();

        assert_eq!(execution.events.len(), 1);
        assert_eq!(execution.events[0].finish_reason, Some(FinishReason::Stop));
        assert!(execution.requests[0].generated_tokens.is_empty());
        assert_eq!(execution.requests[0].remaining_new_tokens(), 0);
    }

    #[test]
    fn mock_executor_preserves_request_identity_and_resets_indices_per_request() {
        let mut first = RequestState::from_parts(
            RequestId::new(13),
            SequenceId::new(14),
            "first".into(),
            vec![1],
            1,
            true,
        )
        .unwrap();
        first.mark_queued().unwrap();
        first.mark_running().unwrap();
        let mut second = RequestState::from_parts(
            RequestId::new(15),
            SequenceId::new(16),
            "second".into(),
            vec![2],
            1,
            true,
        )
        .unwrap();
        second.mark_queued().unwrap();
        second.mark_running().unwrap();
        let executor = MockExecutor::new(MockExecutorConfig::new(vec![7]));

        let execution = executor.execute(vec![first, second]).unwrap();

        assert_eq!(execution.events.len(), 4);
        assert_eq!(execution.events[0].request_id, RequestId::new(13));
        assert_eq!(execution.events[0].sequence_id, SequenceId::new(14));
        assert_eq!(execution.events[0].token_id, Some(7));
        assert_eq!(execution.events[0].index, 0);
        assert_eq!(execution.events[1].request_id, RequestId::new(13));
        assert_eq!(execution.events[1].sequence_id, SequenceId::new(14));
        assert_eq!(
            execution.events[1].finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(execution.events[1].index, 1);
        assert_eq!(execution.events[2].request_id, RequestId::new(15));
        assert_eq!(execution.events[2].sequence_id, SequenceId::new(16));
        assert_eq!(execution.events[2].token_id, Some(7));
        assert_eq!(execution.events[2].index, 0);
        assert_eq!(execution.events[3].request_id, RequestId::new(15));
        assert_eq!(execution.events[3].sequence_id, SequenceId::new(16));
        assert_eq!(
            execution.events[3].finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(execution.events[3].index, 1);
    }

    #[test]
    fn mock_executor_rejects_pending_request_without_panicking() {
        let request = RequestState::from_parts(
            RequestId::new(17),
            SequenceId::new(18),
            "pending".into(),
            vec![1],
            1,
            true,
        )
        .unwrap();
        let executor = MockExecutor::new(MockExecutorConfig::new(vec![7]));

        assert!(executor.execute(vec![request]).is_err());
    }
}
