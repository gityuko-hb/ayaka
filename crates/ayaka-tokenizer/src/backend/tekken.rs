use crate::backend::read_json;
use crate::error::{TokenizerError, TokenizerResult};
use base64::Engine;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use tiktoken_rs::CoreBPE;

/// Default BPE pre-tokenization regex for Tekken (Mistral Nemo, Mistral v3+).
///
/// Matches the cl100k_base / Llama 3 pattern which Mistral Tekken is based on.
/// If `tekken.json` contains an explicit `"pattern"` field, that takes
/// precedence over this constant.
const TEKKEN_DEFAULT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

type TekkenVocab = (HashMap<Vec<u8>, u32>, HashMap<u32, Vec<u8>>);

pub struct TekkenBackend {
    /// CoreBPE for correct rank-based BPE encoding.
    ///
    /// The original implementation used a greedy longest-match scan which
    /// does NOT implement BPE. BPE requires merging pairs in rank order
    /// (lowest rank = highest priority), not taking the longest prefix.
    /// Counter-example: vocab {ab:5, bc:2, cd:3}, input "abcd" →
    ///   BPE  gives [a, bc, d]  (merge bc first because rank 2 < rank 5)
    ///   greedy gives [ab, cd]  (wrong)
    bpe: CoreBPE,
    /// bytes → rank, used for `token_to_id` reverse lookups.
    encoder: HashMap<Vec<u8>, u32>,
    /// rank → bytes, for O(1) `token_bytes` on regular vocab tokens.
    decoder: HashMap<u32, Vec<u8>>,
    /// token string → rank, for `token_to_id` with special tokens.
    special_tokens: HashMap<String, u32>,
    /// Pre-built set for O(1) `is_special_token_id` in the hot path.
    special_ids: HashSet<u32>,
    /// Pre-built reverse map for O(1) `token_bytes` on special tokens.
    special_decoder: HashMap<u32, Vec<u8>>,
}

impl fmt::Debug for TekkenBackend {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("TekkenBackend")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl TekkenBackend {
    pub fn from_path(root: &Path) -> TokenizerResult<Self> {
        let json = read_json(&root.join("tekken.json"))?;

        // Mistral stores `"version": "3"` as a JSON string, not a
        // number. The original `as_u64()` silently returned `None` for string
        // values, `unwrap_or(1)` made the version check never trigger.
        let version: u64 = match json
            .get("version")
            .or_else(|| json.get("schema_version"))
        {
            Some(Value::Number(n)) => n.as_u64().unwrap_or(1),
            Some(Value::String(s)) => s.parse().unwrap_or(1),
            _ => 1,
        };
        if version > 1 {
            return Err(TokenizerError::BackendError(format!(
                "unsupported tekken schema version {version}"
            )));
        }

        // Load vocab and special tokens via refactored associated functions
        // that return data structures rather than mutating &mut self.
        let (encoder, decoder) = Self::collect_vocab(&json)?;
        if decoder.is_empty() {
            return Err(TokenizerError::BackendError(
                "tekken.json does not contain a supported vocab".into(),
            ));
        }

        let special_tokens = Self::collect_special_tokens(&json);

        // Pre-build O(1) lookup structures from special_tokens once.
        let special_ids: HashSet<u32> = special_tokens.values().copied().collect();
        let special_decoder: HashMap<u32, Vec<u8>> = special_tokens
            .iter()
            .map(|(s, &id)| (id, s.as_bytes().to_vec()))
            .collect();

        // Read the regex pattern from the JSON, fall back to default.
        let pattern = json
            .get("pattern")
            .or_else(|| json.pointer("/tokenizer/pattern"))
            .and_then(Value::as_str)
            .unwrap_or(TEKKEN_DEFAULT_PATTERN);

        // Build CoreBPE for correct rank-based BPE algorithm.
        let bpe_encoder: FxHashMap<Vec<u8>, u32> = encoder
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        let bpe_special: FxHashMap<String, u32> = special_tokens
            .iter()
            .map(|(s, &id)| (s.clone(), id))
            .collect();

        let bpe = CoreBPE::new(bpe_encoder, bpe_special, pattern).map_err(|e| {
            TokenizerError::BackendError(format!(
                "failed to build CoreBPE from tekken vocab: {e}. \
                 The tekken.json ranks may not form a valid BPE merge table."
            ))
        })?;

        Ok(Self {
            bpe,
            encoder,
            decoder,
            special_tokens,
            special_ids,
            special_decoder,
        })
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<u32>> {
        // P0 fix: delegate to CoreBPE which implements the correct rank-based
        // BPE merge algorithm. The `as u32` cast is safe because token IDs
        // in any practical vocabulary fit in u32.
        let ids: Vec<u32> = if add_special_tokens {
            self.bpe
                .encode_with_special_tokens(text)
                .into_iter()
                .collect()
        } else {
            self.bpe
                .encode_ordinary(text)
                .into_iter()
                .collect()
        };
        Ok(ids)
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special_tokens && self.special_ids.contains(&id) {
                continue;
            }
            bytes.extend(self.token_bytes(id)?);
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
        // Lookups are O(1) HashMap gets — no linear scan.
        self.decoder
            .get(&id)
            .cloned()
            .or_else(|| self.special_decoder.get(&id).cloned())
            .ok_or(TokenizerError::OutOfVocab(id))
    }

    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        self.special_ids.contains(&id) // Lookups are O(1) HashMap gets — no linear scan.
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        self.special_ids.iter().copied().collect()
    }

    /// Parse the regular vocab section of `tekken.json`.
    ///
    /// Returns `(encoder, decoder)` where `encoder` maps bytes → rank and
    /// `decoder` maps rank → bytes.
    fn collect_vocab(json: &Value) -> TokenizerResult<TekkenVocab> {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        let Some(vocab) = json
            .get("vocab")
            .or_else(|| json.get("tokens"))
            .or_else(|| json.get("model").and_then(|m| m.get("vocab")))
        else {
            return Ok((encoder, decoder));
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
                    Self::insert(&mut encoder, &mut decoder, token.as_bytes().to_vec(), id);
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

                    Self::insert(&mut encoder, &mut decoder, bytes, id);
                }
            },
            _ => {},
        }

        Ok((encoder, decoder))
    }

    /// Parse the special tokens section of `tekken.json`.
    fn collect_special_tokens(json: &Value) -> HashMap<String, u32> {
        let mut out = HashMap::new();

        let Some(special_tokens) = json
            .get("special_tokens")
            .or_else(|| json.get("specialTokens"))
        else {
            return out;
        };

        match special_tokens {
            Value::Object(entries) => {
                for (token, value) in entries {
                    if let Some(id) = value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                    {
                        out.insert(token.clone(), id);
                    } else if let Some(id) = value
                        .get("id")
                        .or_else(|| value.get("rank"))
                        .and_then(Value::as_u64)
                        .and_then(|id| u32::try_from(id).ok())
                    {
                        out.insert(token.clone(), id);
                    }
                }
            },
            Value::Array(entries) => {
                // When ranks are relative offsets, the base is the regular vocab size.
                let base_id: u32 = json
                    .get("vocab")
                    .or_else(|| json.get("tokens"))
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .and_then(|len| u32::try_from(len).ok())
                    .unwrap_or(u32::MAX);

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

                    out.insert(token.to_owned(), id);
                }
            },
            _ => {},
        }

        out
    }

    fn insert(
        encoder: &mut HashMap<Vec<u8>, u32>,
        decoder: &mut HashMap<u32, Vec<u8>>,
        bytes: Vec<u8>,
        id: u32,
    ) {
        encoder.insert(bytes.clone(), id);
        decoder.insert(id, bytes);
    }
}
