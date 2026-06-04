use ayaka_core::id::{RequestId, SequenceId, TokenId};

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum RequestError {
    #[error("prompt_tokens must not be empty")]
    EmptyPromptTokens,

    #[error("max_new_tokens must be greater than zero")]
    EmptyGenerationBudget,

    #[error("request {request_id} cannot transition from {from:?} to {to:?}")]
    InvalidTransition {
        request_id: RequestId,
        from: RequestStatus,
        to: RequestStatus,
    },

    #[error("request {request_id} cannot record generated tokens while {status:?}")]
    InvalidGeneratedTokenState {
        request_id: RequestId,
        status: RequestStatus,
    },

    #[error("request {request_id} exhausted generation budget of {max_new_tokens} tokens")]
    GenerationBudgetExhausted {
        request_id: RequestId,
        max_new_tokens: usize,
    },
}

pub type RequestResult<T> = Result<T, RequestError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RequestStatus {
    Pending,
    Queued,
    Running,
    Completed,
    Rejected,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SequenceState {
    pub id: SequenceId,
    pub request_id: RequestId,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub finished: bool,
}

impl SequenceState {
    pub fn new(
        id: SequenceId,
        request_id: RequestId,
        prompt_token_count: usize,
    ) -> Self {
        Self {
            id,
            request_id,
            prompt_token_count,
            generated_token_count: 0,
            finished: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequestState {
    pub id: RequestId,
    pub prompt: String,
    pub prompt_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub stream: bool,
    pub status: RequestStatus,
    pub sequence: SequenceState,
    pub finish_reason: Option<FinishReason>,
}

impl RequestState {
    pub fn new(
        prompt: String,
        prompt_tokens: Vec<TokenId>,
        max_new_tokens: usize,
        stream: bool,
    ) -> RequestResult<Self> {
        Self::from_parts(
            RequestId::generate(),
            SequenceId::generate(),
            prompt,
            prompt_tokens,
            max_new_tokens,
            stream,
        )
    }

    pub fn from_parts(
        request_id: RequestId,
        sequence_id: SequenceId,
        prompt: String,
        prompt_tokens: Vec<TokenId>,
        max_new_tokens: usize,
        stream: bool,
    ) -> RequestResult<Self> {
        validate_request_parts(&prompt_tokens, max_new_tokens)?;
        Ok(Self {
            id: request_id,
            prompt,
            sequence: SequenceState::new(sequence_id, request_id, prompt_tokens.len()),
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_new_tokens,
            stream,
            status: RequestStatus::Pending,
            finish_reason: None,
        })
    }

    pub fn mark_queued(&mut self) -> RequestResult<()> {
        self.transition(RequestStatus::Queued)
    }

    pub fn mark_running(&mut self) -> RequestResult<()> {
        self.transition(RequestStatus::Running)
    }

    pub fn mark_rejected(&mut self) -> RequestResult<()> {
        self.transition(RequestStatus::Rejected)?;
        self.sequence.finished = true;
        self.finish_reason = Some(FinishReason::Error);
        Ok(())
    }

    pub fn mark_cancelled(&mut self) -> RequestResult<()> {
        self.transition(RequestStatus::Cancelled)?;
        self.sequence.finished = true;
        self.finish_reason = Some(FinishReason::Cancelled);
        Ok(())
    }

    pub fn mark_completed(
        &mut self,
        finish_reason: FinishReason,
    ) -> RequestResult<()> {
        self.transition(RequestStatus::Completed)?;
        self.sequence.finished = true;
        self.finish_reason = Some(finish_reason);
        Ok(())
    }

    pub fn record_generated_token(
        &mut self,
        token_id: TokenId,
    ) -> RequestResult<()> {
        if self.status != RequestStatus::Running {
            return Err(RequestError::InvalidGeneratedTokenState {
                request_id: self.id,
                status: self.status,
            });
        }

        if self.remaining_new_tokens() == 0 {
            return Err(RequestError::GenerationBudgetExhausted {
                request_id: self.id,
                max_new_tokens: self.max_new_tokens,
            });
        }

        self.generated_tokens.push(token_id);
        self.sequence.generated_token_count = self.generated_tokens.len();
        Ok(())
    }

    pub fn remaining_new_tokens(&self) -> usize {
        if matches!(
            self.status,
            RequestStatus::Completed | RequestStatus::Cancelled | RequestStatus::Rejected
        ) {
            return 0;
        }

        self.max_new_tokens
            .saturating_sub(self.generated_tokens.len())
    }

    fn transition(
        &mut self,
        to: RequestStatus,
    ) -> RequestResult<()> {
        let from = self.status;
        let allowed = matches!(
            (from, to),
            (RequestStatus::Pending, RequestStatus::Queued)
                | (RequestStatus::Queued, RequestStatus::Running)
                | (RequestStatus::Running, RequestStatus::Completed)
                | (RequestStatus::Pending, RequestStatus::Rejected)
                | (RequestStatus::Queued, RequestStatus::Rejected)
                | (RequestStatus::Pending, RequestStatus::Cancelled)
                | (RequestStatus::Queued, RequestStatus::Cancelled)
                | (RequestStatus::Running, RequestStatus::Cancelled)
        );

        if allowed {
            self.status = to;
            Ok(())
        } else {
            Err(RequestError::InvalidTransition {
                request_id: self.id,
                from,
                to,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenEvent {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_id: Option<TokenId>,
    pub text_delta: Option<String>,
    pub index: usize,
    pub finish_reason: Option<FinishReason>,
}

impl TokenEvent {
    pub fn token(
        request_id: RequestId,
        sequence_id: SequenceId,
        token_id: TokenId,
        text_delta: Option<String>,
        index: usize,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            token_id: Some(token_id),
            text_delta,
            index,
            finish_reason: None,
        }
    }

    pub fn finished(
        request_id: RequestId,
        sequence_id: SequenceId,
        index: usize,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            token_id: None,
            text_delta: None,
            index,
            finish_reason: Some(finish_reason),
        }
    }
}

pub fn validate_request_parts(
    prompt_tokens: &[TokenId],
    max_new_tokens: usize,
) -> RequestResult<()> {
    if prompt_tokens.is_empty() {
        return Err(RequestError::EmptyPromptTokens);
    }
    if max_new_tokens == 0 {
        return Err(RequestError::EmptyGenerationBudget);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_state_tracks_lifecycle_and_generated_tokens() {
        let request_id = RequestId::new(7);
        let sequence_id = SequenceId::new(11);
        let mut state =
            RequestState::from_parts(request_id, sequence_id, "hello".into(), vec![1, 2], 2, true)
                .unwrap();

        assert_eq!(state.status, RequestStatus::Pending);
        assert_eq!(state.sequence.prompt_token_count, 2);
        assert_eq!(state.remaining_new_tokens(), 2);

        state.mark_queued().unwrap();
        state.mark_running().unwrap();
        state.record_generated_token(42).unwrap();
        state.record_generated_token(43).unwrap();
        state
            .mark_completed(FinishReason::Length)
            .unwrap();

        assert_eq!(state.status, RequestStatus::Completed);
        assert_eq!(state.generated_tokens, vec![42, 43]);
        assert_eq!(state.sequence.generated_token_count, 2);
        assert!(state.sequence.finished);
        assert_eq!(state.finish_reason, Some(FinishReason::Length));
        assert_eq!(state.remaining_new_tokens(), 0);
    }

    #[test]
    fn completed_request_reports_no_remaining_budget_after_early_stop() {
        let mut state = RequestState::from_parts(
            RequestId::new(8),
            SequenceId::new(12),
            "hello".into(),
            vec![1, 2],
            3,
            true,
        )
        .unwrap();

        state.mark_queued().unwrap();
        state.mark_running().unwrap();
        state.record_generated_token(42).unwrap();
        state.mark_completed(FinishReason::Stop).unwrap();

        assert_eq!(state.status, RequestStatus::Completed);
        assert_eq!(state.finish_reason, Some(FinishReason::Stop));
        assert!(state.sequence.finished);
        assert_eq!(state.remaining_new_tokens(), 0);
    }

    #[test]
    fn recording_generated_token_before_running_is_rejected() {
        let request_id = RequestId::new(9);
        let mut state = RequestState::from_parts(
            request_id,
            SequenceId::new(13),
            "hello".into(),
            vec![1],
            1,
            true,
        )
        .unwrap();

        let err = state.record_generated_token(42).unwrap_err();

        assert_eq!(
            err,
            RequestError::InvalidGeneratedTokenState {
                request_id,
                status: RequestStatus::Pending,
            }
        );
        assert!(state.generated_tokens.is_empty());
        assert_eq!(state.sequence.generated_token_count, 0);
    }

    #[test]
    fn recording_generated_token_after_completion_is_rejected() {
        let request_id = RequestId::new(10);
        let mut state = RequestState::from_parts(
            request_id,
            SequenceId::new(14),
            "hello".into(),
            vec![1],
            2,
            true,
        )
        .unwrap();

        state.mark_queued().unwrap();
        state.mark_running().unwrap();
        state.record_generated_token(42).unwrap();
        state.mark_completed(FinishReason::Stop).unwrap();
        let err = state.record_generated_token(43).unwrap_err();

        assert_eq!(
            err,
            RequestError::InvalidGeneratedTokenState {
                request_id,
                status: RequestStatus::Completed,
            }
        );
        assert_eq!(state.generated_tokens, vec![42]);
        assert_eq!(state.sequence.generated_token_count, 1);
    }

    #[test]
    fn recording_generated_token_beyond_budget_is_rejected() {
        let request_id = RequestId::new(11);
        let mut state = RequestState::from_parts(
            request_id,
            SequenceId::new(15),
            "hello".into(),
            vec![1],
            1,
            true,
        )
        .unwrap();

        state.mark_queued().unwrap();
        state.mark_running().unwrap();
        state.record_generated_token(42).unwrap();
        let err = state.record_generated_token(43).unwrap_err();

        assert_eq!(
            err,
            RequestError::GenerationBudgetExhausted {
                request_id,
                max_new_tokens: 1,
            }
        );
        assert_eq!(state.generated_tokens, vec![42]);
        assert_eq!(state.sequence.generated_token_count, 1);
    }

    #[test]
    fn rejected_request_is_terminal_with_error_finish_reason() {
        let mut state = RequestState::from_parts(
            RequestId::new(12),
            SequenceId::new(16),
            "hello".into(),
            vec![1],
            2,
            true,
        )
        .unwrap();

        state.mark_rejected().unwrap();

        assert_eq!(state.status, RequestStatus::Rejected);
        assert!(state.sequence.finished);
        assert_eq!(state.finish_reason, Some(FinishReason::Error));
        assert_eq!(state.remaining_new_tokens(), 0);
    }

    #[test]
    fn validation_rejects_empty_prompt_tokens_and_zero_budget() {
        let empty_prompt = RequestState::from_parts(
            RequestId::new(1),
            SequenceId::new(1),
            "empty".into(),
            Vec::new(),
            1,
            true,
        )
        .unwrap_err();
        assert_eq!(empty_prompt, RequestError::EmptyPromptTokens);

        let zero_budget = RequestState::from_parts(
            RequestId::new(1),
            SequenceId::new(1),
            "hello".into(),
            vec![1],
            0,
            true,
        )
        .unwrap_err();
        assert_eq!(zero_budget, RequestError::EmptyGenerationBudget);
    }

    #[test]
    fn token_events_represent_delta_and_finish_events() {
        let token = TokenEvent::token(
            RequestId::new(3),
            SequenceId::new(5),
            99,
            Some("token:99".into()),
            0,
        );
        assert_eq!(token.token_id, Some(99));
        assert_eq!(token.text_delta.as_deref(), Some("token:99"));
        assert_eq!(token.finish_reason, None);

        let finished =
            TokenEvent::finished(RequestId::new(3), SequenceId::new(5), 1, FinishReason::Stop);
        assert_eq!(finished.token_id, None);
        assert_eq!(finished.index, 1);
        assert_eq!(finished.finish_reason, Some(FinishReason::Stop));
    }
}
