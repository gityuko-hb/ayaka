use crate::backend::read_json;
use crate::error::{TokenizerError, TokenizerResult};
use base64::Engine;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use tiktoken_rs::{CoreBPE, Rank};

/// GPT-2 pre-tokenization regex.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Llama 3 / cl100k_base pre-tokenization regex.
///
/// Used by Llama 3.x, GPT-4, and other models that ship tiktoken-format
/// vocabularies. Differs from GPT-2 in numeric splitting and case-insensitive
/// contraction handling — using GPT2_PATTERN for these models produces
/// tokenizations that diverge from the reference implementation.
const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct TiktokenBackend {
    bpe: CoreBPE,
    decoder: HashMap<u32, Vec<u8>>,
    encoder: HashMap<Vec<u8>, u32>,
    special_tokens: HashMap<String, u32>,
    /// Pre-built set for O(1) `is_special_token_id` checks in the hot path.
    special_ids: HashSet<u32>,
    /// Pre-built reverse map for O(1) `token_bytes` on special tokens.
    special_decoder: HashMap<u32, Vec<u8>>,
}

impl fmt::Debug for TiktokenBackend {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("TiktokenBackend")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl TiktokenBackend {
    pub fn from_path(root: &Path) -> TokenizerResult<Self> {
        let mut encoder = FxHashMap::<Vec<u8>, Rank>::default();
        let mut decoder = HashMap::new();
        let raw = std::fs::read_to_string(root.join("tokenizer.model"))?;

        for (line_no, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token = parts
                .next()
                .ok_or_else(|| malformed_line(line_no))?;
            let rank = parts
                .next()
                .ok_or_else(|| malformed_line(line_no))?
                .parse::<u32>()
                .map_err(|_| malformed_line(line_no))?;
            if parts.next().is_some() {
                return Err(malformed_line(line_no));
            }
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(token)
                .map_err(|err| TokenizerError::BackendError(err.to_string()))?;
            encoder.insert(bytes.clone(), rank);
            decoder.insert(rank, bytes);
        }

        let special_tokens = special_tokens(root)?;
        let special_encoder: FxHashMap<String, Rank> = special_tokens
            .iter()
            .map(|(token, id)| (token.clone(), *id))
            .collect();

        // Load pattern from tokenizer_config.json instead of hardcoding
        // GPT-2 pattern. Different tiktoken models use different regex patterns,
        // and using the wrong one produces tokenizations that diverge from the
        // HuggingFace reference implementation.
        let pattern = load_pattern(root);

        let bpe = CoreBPE::new(encoder.clone(), special_encoder, &pattern)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))?;

        // Pre-build O(1) lookup structures so the generation hot path
        // never does an O(n) linear scan over special_tokens.
        let special_ids: HashSet<u32> = special_tokens.values().copied().collect();
        let special_decoder: HashMap<u32, Vec<u8>> = special_tokens
            .iter()
            .map(|(s, &id)| (id, s.as_bytes().to_vec()))
            .collect();

        Ok(Self {
            bpe,
            decoder,
            encoder: encoder.into_iter().collect(),
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
        if add_special_tokens {
            Ok(self.bpe.encode_with_special_tokens(text))
        } else {
            Ok(self.bpe.encode_ordinary(text))
        }
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        let filtered: Vec<u32> = if skip_special_tokens {
            ids.iter()
                .copied()
                .filter(|id| !self.special_ids.contains(id)) // P1: O(1)
                .collect()
        } else {
            ids.to_vec()
        };
        self.bpe
            .decode(&filtered)
            .map_err(|err| TokenizerError::BackendError(err.to_string()))
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
            .or_else(|| self.special_decoder.get(&id).cloned()) // P1: O(1)
            .ok_or(TokenizerError::OutOfVocab(id))
    }

    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        self.special_ids.contains(&id) // P1: O(1)
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        self.special_ids.iter().copied().collect()
    }
}

/// Load the BPE pre-tokenization regex from `tokenizer_config.json`.
///
/// Priority:
/// 1. Explicit `"pattern"` field in the config.
/// 2. Inferred from `"tokenizer_class"`:
///    - any class containing `"Llama"` → Llama 3 / cl100k_base pattern.
/// 3. GPT-2 pattern as a safe default.
fn load_pattern(root: &Path) -> String {
    let Ok(config) = read_json(&root.join("tokenizer_config.json")) else {
        return GPT2_PATTERN.to_owned();
    };

    // Explicit pattern takes precedence.
    if let Some(pattern) = config.get("pattern").and_then(Value::as_str) {
        return pattern.to_owned();
    }

    // Fall back to inference from tokenizer class name.
    match config
        .get("tokenizer_class")
        .and_then(Value::as_str)
    {
        Some(class) if class.contains("Llama") => LLAMA3_PATTERN.to_owned(),
        _ => GPT2_PATTERN.to_owned(),
    }
}

fn malformed_line(line_no: usize) -> TokenizerError {
    TokenizerError::BackendError(format!("invalid tiktoken model line {}", line_no + 1))
}

fn special_tokens(root: &Path) -> TokenizerResult<HashMap<String, u32>> {
    let config_path = root.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(HashMap::new());
    }

    let config = read_json(&config_path)?;
    let mut out = HashMap::new();
    collect_config_token(&config, "bos_token", "bos_token_id", &mut out);
    collect_config_token(&config, "eos_token", "eos_token_id", &mut out);
    collect_config_token(&config, "pad_token", "pad_token_id", &mut out);
    collect_config_token(&config, "unk_token", "unk_token_id", &mut out);

    if let (Some(Value::Array(tokens)), Some(Value::Array(ids))) = (
        config.get("additional_special_tokens"),
        config.get("additional_special_tokens_ids"),
    ) {
        for (token, id) in tokens.iter().zip(ids) {
            if let (Some(token), Some(id)) = (token.as_str(), id.as_u64())
                && let Ok(id) = u32::try_from(id)
            {
                out.insert(token.to_owned(), id);
            }
        }
    }

    Ok(out)
}

fn collect_config_token(
    config: &Value,
    token_key: &str,
    id_key: &str,
    out: &mut HashMap<String, u32>,
) {
    let token = match config.get(token_key) {
        Some(Value::String(token)) => Some(token.as_str()),
        Some(Value::Object(obj)) => obj.get("content").and_then(Value::as_str),
        _ => None,
    };
    let id = config
        .get(id_key)
        .and_then(|value| match value {
            Value::Number(n) => n.as_u64(),
            Value::Object(obj) => obj.get("id").and_then(Value::as_u64),
            _ => None,
        })
        .and_then(|id| u32::try_from(id).ok());

    if let (Some(token), Some(id)) = (token, id) {
        out.insert(token.to_owned(), id);
    }
}
