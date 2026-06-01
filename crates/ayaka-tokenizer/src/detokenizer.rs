use crate::backend::TokenizerBackend;
use crate::error::{TokenizerError, TokenizerResult};
use std::sync::Arc;

#[derive(Debug)]
pub struct StreamingDetokenizer {
    tokenizer: Arc<TokenizerBackend>,
    byte_buffer: Vec<u8>,
    skip_special_tokens: bool,
    special_token_ids: std::collections::HashSet<u32>,
    is_first_token: bool,
}

impl StreamingDetokenizer {
    pub(crate) fn new(
        tokenizer: Arc<TokenizerBackend>,
        skip_special_tokens: bool,
        special_token_ids: std::collections::HashSet<u32>,
    ) -> Self {
        Self {
            tokenizer,
            byte_buffer: Vec::new(),
            skip_special_tokens,
            special_token_ids,
            is_first_token: true,
        }
    }

    pub fn step(
        &mut self,
        token_id: u32,
    ) -> TokenizerResult<Option<String>> {
        if token_id as usize >= self.tokenizer.vocab_size() {
            return Err(TokenizerError::OutOfVocab(token_id));
        }

        if self.skip_special_tokens
            && (self.special_token_ids.contains(&token_id)
                || self.tokenizer.is_special_token_id(token_id))
        {
            self.is_first_token = false;
            return Ok(None);
        }

        self.byte_buffer
            .extend(self.tokenizer.token_bytes(token_id)?);
        self.is_first_token = false;

        match std::str::from_utf8(&self.byte_buffer) {
            Ok(text) => {
                let chunk = text.to_owned();
                self.byte_buffer.clear();
                Ok((!chunk.is_empty()).then_some(chunk))
            },
            Err(err) if err.error_len().is_none() => Ok(None),
            Err(_) => Err(TokenizerError::InvalidUtf8),
        }
    }

    pub fn flush(&mut self) -> Option<String> {
        if self.byte_buffer.is_empty() {
            return None;
        }

        let tail = String::from_utf8_lossy(&self.byte_buffer).into_owned();
        self.byte_buffer.clear();
        (!tail.is_empty()).then_some(tail)
    }

    pub fn reset(&mut self) {
        self.byte_buffer.clear();
        self.is_first_token = true;
    }
}
