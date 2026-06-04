use ayaka_core::id::TokenId;
use ayaka_request::{RequestError, RequestState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateRequest {
    pub prompt: String,
    pub prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub stream: bool,
}

impl GenerateRequest {
    pub fn new(
        prompt: impl Into<String>,
        prompt_tokens: Vec<TokenId>,
        max_new_tokens: usize,
        stream: bool,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            prompt_tokens,
            max_new_tokens,
            stream,
        }
    }

    pub fn into_request_state(self) -> Result<RequestState, RequestError> {
        RequestState::new(
            self.prompt,
            self.prompt_tokens,
            self.max_new_tokens,
            self.stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ayaka_request::RequestError;

    #[test]
    fn generate_request_converts_to_internal_request_state() {
        let request = GenerateRequest::new("hello", vec![1, 2], 3, true);
        let state = request.into_request_state().unwrap();

        assert_eq!(state.prompt, "hello");
        assert_eq!(state.prompt_tokens, vec![1, 2]);
        assert_eq!(state.max_new_tokens, 3);
        assert!(state.stream);
    }

    #[test]
    fn generate_request_rejects_empty_tokens() {
        let err = GenerateRequest::new("hello", Vec::new(), 3, true)
            .into_request_state()
            .unwrap_err();

        assert_eq!(err, RequestError::EmptyPromptTokens);
    }

    #[test]
    fn generate_request_rejects_zero_budget() {
        let err = GenerateRequest::new("hello", vec![1], 0, false)
            .into_request_state()
            .unwrap_err();

        assert_eq!(err, RequestError::EmptyGenerationBudget);
    }
}
