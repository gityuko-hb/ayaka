use crate::backend::TokenizerBackend;
use crate::error::{TokenizerError, TokenizerResult};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug)]
pub struct StreamingDetokenizer {
    tokenizer: Arc<TokenizerBackend>,
    byte_buffer: Vec<u8>,
    skip_special_tokens: bool,
    special_token_ids: HashSet<u32>,
    is_first_token: bool,
    /// Whether to strip a leading space (0x20) from the first non-special token.
    ///
    /// SentencePiece models encode word boundaries as `▁`, which `token_bytes`
    /// converts to 0x20. The first real token after BOS always carries this
    /// prefix even though no space precedes the start of the generation — this
    /// flag corrects the artifact by stripping it.
    ///
    /// Set to `true` only for the `SentencePiece` backend; `false` for
    /// `HuggingFace`, `Tiktoken`, and `Tekken` where leading spaces are
    /// semantically meaningful.
    strip_leading_space: bool,
}

impl StreamingDetokenizer {
    pub(crate) fn new(
        tokenizer: Arc<TokenizerBackend>,
        skip_special_tokens: bool,
        special_token_ids: HashSet<u32>,
        strip_leading_space: bool,
    ) -> Self {
        Self {
            tokenizer,
            byte_buffer: Vec::new(),
            skip_special_tokens,
            special_token_ids,
            is_first_token: true,
            strip_leading_space,
        }
    }

    /// Process one token from the sampler.
    ///
    /// Returns `Some(chunk)` when the internal byte buffer forms a complete
    /// UTF-8 sequence, `None` when more tokens are needed to complete a
    /// multi-byte codepoint.
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
            // Special tokens (BOS, EOS, PAD, …) do NOT advance `is_first_token`
            // so that the leading-space stripping still applies to the first
            // real word even when BOS is present in the stream.
            return Ok(None);
        }

        let mut bytes = self.tokenizer.token_bytes(token_id)?;

        // P0 fix: strip the ▁-derived leading space from the first non-special
        // token in SentencePiece models. For tiktoken / tekken / HF models
        // `strip_leading_space` is false, so this branch is never taken.
        if self.is_first_token && self.strip_leading_space {
            if bytes.first() == Some(&b' ') {
                bytes.remove(0);
            }
        }

        self.is_first_token = false;
        self.byte_buffer.extend(bytes);

        match std::str::from_utf8(&self.byte_buffer) {
            Ok(text) => {
                let chunk = text.to_owned();
                self.byte_buffer.clear();
                Ok((!chunk.is_empty()).then_some(chunk))
            },
            // Incomplete multi-byte sequence — wait for the next token.
            Err(err) if err.error_len().is_none() => Ok(None),
            // Genuine invalid UTF-8 — the byte sequence can never be valid.
            Err(_) => Err(TokenizerError::InvalidUtf8),
        }
    }

    /// Flush any bytes remaining in the buffer at end-of-sequence.
    ///
    /// Invalid UTF-8 sequences are replaced with U+FFFD rather than panicking,
    /// so this method is safe to call unconditionally after the generation loop.
    pub fn flush(&mut self) -> Option<String> {
        if self.byte_buffer.is_empty() {
            return None;
        }
        let tail = String::from_utf8_lossy(&self.byte_buffer).into_owned();
        self.byte_buffer.clear();
        (!tail.is_empty()).then_some(tail)
    }

    /// Reset state for the next request without re-allocating heap storage.
    pub fn reset(&mut self) {
        self.byte_buffer.clear();
        self.is_first_token = true;
    }
}
