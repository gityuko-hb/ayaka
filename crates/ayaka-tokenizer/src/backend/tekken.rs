use crate::backend::read_json;
use crate::error::{TokenizerError, TokenizerResult};
use base64::Engine;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub struct TekkenBackend {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    special_tokens: HashMap<String, u32>,
}

impl TekkenBackend {
    pub fn from_path(root: &Path) -> TokenizerResult<Self> {
        let json = read_json(&root.join("tekken.json"))?;
        let version = json
            .get("version")
            .or_else(|| json.get("schema_version"))
            .and_then(Value::as_u64)
            .unwrap_or(1);
        if version > 1 {
            return Err(TokenizerError::BackendError(format!(
                "unsupported tekken schema version {version}"
            )));
        }

        let mut backend = Self {
            encoder: HashMap::new(),
            decoder: HashMap::new(),
            special_tokens: HashMap::new(),
        };

        backend.load_vocab(&json)?;
        backend.load_special_tokens(&json);

        if backend.decoder.is_empty() {
            return Err(TokenizerError::BackendError(
                "tekken.json does not contain a supported vocab".into(),
            ));
        }

        Ok(backend)
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<u32>> {
        let mut ids = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let mut matched = None;
            for end in ((pos + 1)..=bytes.len()).rev() {
                if let Some(id) = self.encoder.get(&bytes[pos..end]) {
                    matched = Some((*id, end));
                    break;
                }
            }

            let Some((id, end)) = matched else {
                return Err(TokenizerError::BackendError(format!(
                    "tekken vocab cannot encode byte 0x{:02x}",
                    bytes[pos]
                )));
            };

            ids.push(id);
            pos = end;
        }

        if add_special_tokens {
            if let Some(id) = self.special_tokens.get("<s>") {
                ids.insert(0, *id);
            }
            if let Some(id) = self.special_tokens.get("</s>") {
                ids.push(*id);
            }
        }

        Ok(ids)
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        let mut bytes = Vec::new();
        for id in ids {
            if skip_special_tokens && self.is_special_token_id(*id) {
                continue;
            }
            bytes.extend(self.token_bytes(*id)?);
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.special_tokens
            .get(token)
            .copied()
            .or_else(|| self.encoder.get(token.as_bytes()).copied())
    }

    pub fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.token_bytes(id)
            .ok()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len() + self.special_tokens.len()
    }

    pub fn token_bytes(
        &self,
        id: u32,
    ) -> TokenizerResult<Vec<u8>> {
        self.decoder
            .get(&id)
            .cloned()
            .or_else(|| {
                self.special_tokens
                    .iter()
                    .find_map(|(token, token_id)| {
                        (*token_id == id).then(|| token.as_bytes().to_vec())
                    })
            })
            .ok_or(TokenizerError::OutOfVocab(id))
    }

    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        self.special_tokens
            .values()
            .any(|token_id| *token_id == id)
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        self.special_tokens.values().copied().collect()
    }

    fn load_vocab(
        &mut self,
        json: &Value,
    ) -> TokenizerResult<()> {
        let Some(vocab) = json
            .get("vocab")
            .or_else(|| json.get("tokens"))
            .or_else(|| {
                json.get("model")
                    .and_then(|model| model.get("vocab"))
            })
        else {
            return Ok(());
        };

        match vocab {
            Value::Object(entries) => {
                for (token, value) in entries {
                    let Some(id) = value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                    else {
                        continue;
                    };
                    self.insert_token(token.as_bytes().to_vec(), id);
                }
            },
            Value::Array(entries) => {
                for entry in entries {
                    let Some(id) = entry
                        .get("rank")
                        .or_else(|| entry.get("id"))
                        .and_then(Value::as_u64)
                        .and_then(|id| u32::try_from(id).ok())
                    else {
                        continue;
                    };
                    let bytes = if let Some(raw) = entry
                        .get("bytes")
                        .or_else(|| entry.get("token_bytes"))
                        .and_then(Value::as_str)
                    {
                        base64::engine::general_purpose::STANDARD
                            .decode(raw)
                            .map_err(|err| TokenizerError::BackendError(err.to_string()))?
                    } else if let Some(token) = entry
                        .get("token")
                        .or_else(|| entry.get("content"))
                        .or_else(|| entry.get("token_str"))
                        .and_then(Value::as_str)
                    {
                        token.as_bytes().to_vec()
                    } else {
                        continue;
                    };
                    self.insert_token(bytes, id);
                }
            },
            _ => {},
        }

        Ok(())
    }

    fn load_special_tokens(
        &mut self,
        json: &Value,
    ) {
        let Some(special_tokens) = json
            .get("special_tokens")
            .or_else(|| json.get("specialTokens"))
        else {
            return;
        };

        match special_tokens {
            Value::Object(entries) => {
                for (token, value) in entries {
                    if let Some(id) = value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                    {
                        self.special_tokens.insert(token.clone(), id);
                    } else if let Some(id) = value
                        .get("id")
                        .or_else(|| value.get("rank"))
                        .and_then(Value::as_u64)
                        .and_then(|id| u32::try_from(id).ok())
                    {
                        self.special_tokens.insert(token.clone(), id);
                    }
                }
            },
            Value::Array(entries) => {
                let base_id = u32::try_from(self.decoder.len()).unwrap_or(u32::MAX);

                for entry in entries {
                    let Some(token) = entry
                        .get("token")
                        .or_else(|| entry.get("content"))
                        .or_else(|| entry.get("token_str"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };

                    let Some(id) = entry
                        .get("id")
                        .and_then(Value::as_u64)
                        .and_then(|id| u32::try_from(id).ok())
                        .or_else(|| {
                            entry
                                .get("rank")
                                .and_then(Value::as_u64)
                                .and_then(|rank| u32::try_from(rank).ok())
                                .and_then(|rank| base_id.checked_add(rank))
                        })
                    else {
                        continue;
                    };

                    self.special_tokens.insert(token.to_owned(), id);
                }
            },
            _ => {},
        }
    }

    fn insert_token(
        &mut self,
        bytes: Vec<u8>,
        id: u32,
    ) {
        self.encoder.insert(bytes.clone(), id);
        self.decoder.insert(id, bytes);
    }
}
