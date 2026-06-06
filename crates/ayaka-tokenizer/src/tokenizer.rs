use crate::backend::{TokenizerBackend, read_json, tokenizer_root};
use crate::chat_template::{ChatMessage, ChatTemplateRenderer};
use crate::detokenizer::StreamingDetokenizer;
use crate::error::{TokenizerError, TokenizerResult};
use crate::vocab::VocabMetadata;
use rayon::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "hub-download")]
use crate::tokens::{TokenSource, get_token, resolve_default_token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    pub ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct AyakaTokenizer {
    backend: Arc<TokenizerBackend>,
    metadata: VocabMetadata,
    chat_template: Option<ChatTemplateRenderer>,
}

impl AyakaTokenizer {
    pub fn from_path(path: &Path) -> TokenizerResult<Self> {
        let root = tokenizer_root(path);
        let backend = Arc::new(TokenizerBackend::detect_and_load(path)?);
        let config = optional_json(root.join("tokenizer_config.json"))?;
        let mut metadata = VocabMetadata::from_config_value(config.as_ref(), backend.vocab_size());
        metadata.merge_backend_special_ids(backend.special_token_ids());
        let chat_template = ChatTemplateRenderer::from_config_value(config.as_ref())?;

        Ok(Self {
            backend,
            metadata,
            chat_template,
        })
    }

    #[cfg(feature = "hub-download")]
    pub async fn from_pretrained(
        model_id: &str,
        token: Option<&str>,
    ) -> TokenizerResult<Self> {
        let source = resolve_default_token(token)
            .map(TokenSource::Literal)
            .unwrap_or(TokenSource::None);
        Self::from_pretrained_with_source(model_id, source).await
    }

    #[cfg(feature = "hub-download")]
    pub async fn from_pretrained_with_source(
        model_id: &str,
        source: TokenSource,
    ) -> TokenizerResult<Self> {
        use hf_hub::api::tokio::ApiBuilder;

        let api = ApiBuilder::new()
            .with_token(get_token(&source))
            .build()
            .map_err(|err| TokenizerError::HubError(err.to_string()))?;
        let repo = api.model(model_id.to_owned());
        let mut downloaded = Vec::new();

        for file in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tekken.json",
        ] {
            if let Ok(path) = repo.get(file).await {
                downloaded.push(path);
            }
        }

        let Some(root) = common_parent(&downloaded) else {
            return Err(TokenizerError::UnknownFormat(PathBuf::from(model_id)));
        };

        Self::from_path(&root)
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> TokenizerResult<Encoding> {
        self.backend
            .encode(text, add_special_tokens)
            .map(|ids| Encoding { ids })
    }

    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> TokenizerResult<Vec<Encoding>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    pub fn count_tokens(
        &self,
        text: &str,
    ) -> TokenizerResult<usize> {
        self.encode(text, false).map(|enc| enc.ids.len())
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> TokenizerResult<String> {
        self.backend.decode(ids, skip_special_tokens)
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> TokenizerResult<String> {
        self.chat_template
            .as_ref()
            .ok_or(TokenizerError::NoChatTemplate)?
            .render(messages, add_generation_prompt)
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
    ) -> TokenizerResult<Encoding> {
        let prompt = self.apply_chat_template(messages, true)?;
        self.encode(&prompt, false)
    }

    /// Create a per-request streaming detokenizer.
    ///
    /// The `strip_leading_space` flag is forwarded from the backend so that
    /// SentencePiece models correctly strip the `▁`-derived leading space from
    /// the first generated token (P0 fix). Callers do not need to specify this —
    /// it is determined automatically from the loaded tokenizer format.
    pub fn create_detokenizer(
        &self,
        skip_special_tokens: bool,
    ) -> StreamingDetokenizer {
        StreamingDetokenizer::new(
            Arc::clone(&self.backend),
            skip_special_tokens,
            self.metadata.special_token_ids.clone(),
            self.backend.is_space_prefix_tokenizer(),
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.metadata.bos_token_id
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.metadata.eos_token_id
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        self.metadata.pad_token_id
    }

    pub fn unk_token_id(&self) -> Option<u32> {
        self.metadata.unk_token_id
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<u32> {
        self.backend.token_to_id(token)
    }

    pub fn id_to_token(
        &self,
        id: u32,
    ) -> Option<String> {
        self.backend.id_to_token(id)
    }
}

fn optional_json(path: PathBuf) -> TokenizerResult<Option<Value>> {
    if path.exists() {
        read_json(&path).map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(feature = "hub-download")]
fn common_parent(paths: &[PathBuf]) -> Option<PathBuf> {
    paths.first()?.parent().map(Path::to_path_buf)
}
