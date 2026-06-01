use crate::error::{TokenizerError, TokenizerResult};
use std::path::Path;

#[derive(Debug)]
pub struct HuggingFaceBackend {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceBackend {
    pub fn from_path(root: &Path) -> TokenizerResult<Self> {
        let path = if root.is_file() {
            root.to_path_buf()
        } else {
            root.join("tokenizer.json")
        };

        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|err| TokenizerError::BackendError(err.to_string()))
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn token_bytes(
        &self,
        id: u32,
    ) -> TokenizerResult<Vec<u8>> {
        if let Some(token) = self.id_to_token(id) {
            if let Some(byte) = parse_byte_fallback_token(&token) {
                return Ok(vec![byte]);
            }
        }

        self.decode(&[id], false).map(String::into_bytes)
    }

    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        self.id_to_token(id)
            .map(|token| {
                self.tokenizer
                    .get_added_vocabulary()
                    .is_special_token(&token)
            })
            .unwrap_or(false)
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        self.tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter_map(|(id, token)| token.special.then_some(*id))
            .collect()
    }
}

fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}
