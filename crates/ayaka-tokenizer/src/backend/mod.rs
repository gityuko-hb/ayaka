use crate::error::{TokenizerError, TokenizerResult};
use base64::Engine;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[cfg(feature = "backend-hf")]
pub mod huggingface;
#[cfg(feature = "backend-spm")]
pub mod sentencepiece;
#[cfg(feature = "backend-tekken")]
pub mod tekken;
#[cfg(feature = "backend-tiktoken")]
pub mod tiktoken;

#[derive(Debug)]
pub enum TokenizerBackend {
    #[cfg(feature = "backend-hf")]
    HuggingFace(Box<huggingface::HuggingFaceBackend>),
    #[cfg(feature = "backend-spm")]
    SentencePiece(sentencepiece::SentencePieceBackend),
    #[cfg(feature = "backend-tiktoken")]
    Tiktoken(tiktoken::TiktokenBackend),
    #[cfg(feature = "backend-tekken")]
    Tekken(tekken::TekkenBackend),
}

impl TokenizerBackend {
    pub fn detect_and_load(path: &Path) -> TokenizerResult<Self> {
        if path.is_file() {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();

            if file_name.eq_ignore_ascii_case("tekken.json") {
                let root = path.parent().unwrap_or_else(|| Path::new("."));
                return load_tekken(root);
            }

            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
            {
                return load_huggingface(path);
            }
        }

        let root = tokenizer_root(path);

        let tokenizer_json = root.join("tokenizer.json");
        if tokenizer_json.exists() {
            return load_huggingface(&root);
        }

        let tekken_json = root.join("tekken.json");
        if tekken_json.exists() {
            return load_tekken(&root);
        }

        let config_path = root.join("tokenizer_config.json");
        if let Ok(config) = read_json(&config_path)
            && let Some(class) = config
                .get("tokenizer_class")
                .and_then(Value::as_str)
        {
            if class.contains("Mistral") {
                return load_tekken(&root);
            }
            if class.contains("Llama") || class.contains("Gemma") {
                return load_sentencepiece(&root);
            }
            if class.contains("PreTrainedTokenizerFast") {
                return load_huggingface(&root);
            }
        }

        let model_path = root.join("tokenizer.model");
        if model_path.exists() {
            let bytes = std::fs::read(&model_path)?;
            if bytes.first() == Some(&0x0A) {
                return load_sentencepiece(&root);
            }
            if looks_like_tiktoken_model(&bytes) {
                return load_tiktoken(&root);
            }
        }

        Err(TokenizerError::UnknownFormat(root))
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<u32>> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.encode(text, add_special_tokens),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.encode(text, add_special_tokens),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.encode(text, add_special_tokens),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.encode(text, add_special_tokens),
        }
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.decode(ids, skip_special_tokens),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.decode(ids, skip_special_tokens),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.decode(ids, skip_special_tokens),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.decode(ids, skip_special_tokens),
        }
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.token_to_id(token),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.token_to_id(token),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.token_to_id(token),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.token_to_id(token),
        }
    }

    pub fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.id_to_token(id),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.id_to_token(id),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.id_to_token(id),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.id_to_token(id),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.vocab_size(),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.vocab_size(),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.vocab_size(),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.vocab_size(),
        }
    }

    pub fn token_bytes(
        &self,
        id: u32,
    ) -> TokenizerResult<Vec<u8>> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.token_bytes(id),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.token_bytes(id),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.token_bytes(id),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.token_bytes(id),
        }
    }

    pub fn is_special_token_id(
        &self,
        id: u32,
    ) -> bool {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.is_special_token_id(id),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.is_special_token_id(id),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.is_special_token_id(id),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.is_special_token_id(id),
        }
    }

    pub fn special_token_ids(&self) -> Vec<u32> {
        match self {
            #[cfg(feature = "backend-hf")]
            Self::HuggingFace(backend) => backend.special_token_ids(),
            #[cfg(feature = "backend-spm")]
            Self::SentencePiece(backend) => backend.special_token_ids(),
            #[cfg(feature = "backend-tiktoken")]
            Self::Tiktoken(backend) => backend.special_token_ids(),
            #[cfg(feature = "backend-tekken")]
            Self::Tekken(backend) => backend.special_token_ids(),
        }
    }
}

pub(crate) fn tokenizer_root(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    }
}

pub(crate) fn read_json(path: &Path) -> TokenizerResult<Value> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn looks_like_tiktoken_model(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };

    text.lines()
        .filter(|line| !line.trim().is_empty())
        .all(|line| {
            let mut parts = line.split_whitespace();
            let Some(raw) = parts.next() else {
                return false;
            };
            let Some(rank) = parts.next() else {
                return false;
            };

            parts.next().is_none()
                && rank.parse::<u32>().is_ok()
                && base64::engine::general_purpose::STANDARD
                    .decode(raw)
                    .is_ok()
        })
}

#[allow(dead_code)]
fn disabled_backend(name: &str) -> TokenizerError {
    TokenizerError::backend(format!("{name} backend is disabled by feature flags"))
}

fn load_huggingface(root: &Path) -> TokenizerResult<TokenizerBackend> {
    #[cfg(feature = "backend-hf")]
    {
        huggingface::HuggingFaceBackend::from_path(root)
            .map(|b| TokenizerBackend::HuggingFace(Box::new(b)))
    }
    #[cfg(not(feature = "backend-hf"))]
    {
        let _ = root;
        Err(disabled_backend("backend-hf"))
    }
}

fn load_sentencepiece(root: &Path) -> TokenizerResult<TokenizerBackend> {
    #[cfg(feature = "backend-spm")]
    {
        sentencepiece::SentencePieceBackend::from_path(root).map(TokenizerBackend::SentencePiece)
    }
    #[cfg(not(feature = "backend-spm"))]
    {
        let _ = root;
        Err(disabled_backend("backend-spm"))
    }
}

fn load_tiktoken(root: &Path) -> TokenizerResult<TokenizerBackend> {
    #[cfg(feature = "backend-tiktoken")]
    {
        tiktoken::TiktokenBackend::from_path(root).map(TokenizerBackend::Tiktoken)
    }
    #[cfg(not(feature = "backend-tiktoken"))]
    {
        let _ = root;
        Err(disabled_backend("backend-tiktoken"))
    }
}

fn load_tekken(root: &Path) -> TokenizerResult<TokenizerBackend> {
    #[cfg(feature = "backend-tekken")]
    {
        tekken::TekkenBackend::from_path(root).map(TokenizerBackend::Tekken)
    }
    #[cfg(not(feature = "backend-tekken"))]
    {
        let _ = root;
        Err(disabled_backend("backend-tekken"))
    }
}
