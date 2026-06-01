use ayaka_tokenizer::backend::TokenizerBackend;
use ayaka_tokenizer::{
    AyakaTokenizer, ChatMessage, ChatTemplateRenderer, TokenSource, TokenizerError, VocabMetadata,
    get_token,
};
use serde_json::json;
use std::path::Path;

fn write_json(
    path: impl AsRef<Path>,
    value: serde_json::Value,
) {
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}

fn hf_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_json(
        dir.path().join("tokenizer.json"),
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 3, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 4, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "hello": 1,
                    "world": 2,
                    "<s>": 3,
                    "</s>": 4,
                    "user": 5,
                    "assistant": 6,
                    ":": 7
                },
                "unk_token": "[UNK]"
            }
        }),
    );
    write_json(
        dir.path().join("tokenizer_config.json"),
        json!({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": {"id": 3, "content": "<s>"},
            "bos_token_id": 3,
            "eos_token": {"id": 4, "content": "</s>"},
            "eos_token_id": 4,
            "unk_token_id": 0,
            "all_special_ids": [0, 3, 4],
            "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        }),
    );
    dir
}

#[test]
fn token_literal_is_trimmed() {
    assert_eq!(
        get_token(&TokenSource::Literal("  hf_literal  ".into())),
        Some("hf_literal".into())
    );
}

#[test]
fn token_env_var_reads_named_variable() {
    unsafe {
        std::env::set_var("AYAKA_TOKENIZER_TEST_TOKEN", " hf_env ");
    }
    assert_eq!(
        get_token(&TokenSource::EnvVar("AYAKA_TOKENIZER_TEST_TOKEN".into())),
        Some("hf_env".into())
    );
    unsafe {
        std::env::remove_var("AYAKA_TOKENIZER_TEST_TOKEN");
    }
}

#[test]
fn token_missing_cache_returns_none() {
    assert_eq!(
        get_token(&TokenSource::Path("missing-token-file".into())),
        None
    );
}

#[test]
fn vocab_metadata_extracts_special_ids() {
    let metadata = VocabMetadata::from_config_value(
        Some(&json!({
            "bos_token_id": {"id": 10, "content": "<s>"},
            "eos_token_id": 11,
            "pad_token_id": null,
            "unk_token_id": 0,
            "additional_special_tokens_ids": [12, {"id": 13, "content": "<x>"}],
            "added_tokens_decoder": {
                "14": {"content": "<tool>", "special": true},
                "15": {"content": "normal", "special": false}
            }
        })),
        16,
    );

    assert_eq!(metadata.vocab_size, 16);
    assert_eq!(metadata.bos_token_id, Some(10));
    assert_eq!(metadata.eos_token_id, Some(11));
    assert!(metadata.special_token_ids.contains(&14));
    assert!(!metadata.special_token_ids.contains(&15));
}

#[test]
#[cfg(feature = "backend-hf")]
fn backend_detects_huggingface_fixture() {
    let fixture = hf_fixture();
    let backend = TokenizerBackend::detect_and_load(fixture.path()).unwrap();
    assert_eq!(backend.vocab_size(), 8);
    assert_eq!(backend.token_to_id("hello"), Some(1));
    assert_eq!(backend.id_to_token(2), Some("world".into()));
}

#[test]
fn backend_unknown_directory_returns_unknown_format() {
    let dir = tempfile::tempdir().unwrap();
    let err = TokenizerBackend::detect_and_load(dir.path()).unwrap_err();
    assert!(matches!(err, TokenizerError::UnknownFormat(_)));
}

#[test]
#[cfg(feature = "backend-hf")]
fn facade_loads_hf_fixture_and_encodes_text() {
    fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
    assert_send_sync_clone::<AyakaTokenizer>();

    let fixture = hf_fixture();
    let tokenizer = AyakaTokenizer::from_path(fixture.path()).unwrap();

    let encoding = tokenizer.encode("hello world", false).unwrap();
    assert_eq!(encoding.ids, vec![1, 2]);
    assert_eq!(
        tokenizer.decode(&encoding.ids, true).unwrap(),
        "hello world"
    );
    assert_eq!(tokenizer.count_tokens("hello world").unwrap(), 2);
    assert_eq!(tokenizer.vocab_size(), 8);
    assert_eq!(tokenizer.bos_token_id(), Some(3));
    assert_eq!(tokenizer.eos_token_id(), Some(4));
    assert_eq!(tokenizer.unk_token_id(), Some(0));
}

#[test]
#[cfg(feature = "backend-hf")]
fn facade_encode_batch_preserves_order() {
    let fixture = hf_fixture();
    let tokenizer = AyakaTokenizer::from_path(fixture.path()).unwrap();
    let encodings = tokenizer
        .encode_batch(&["hello", "world"], false)
        .unwrap();

    assert_eq!(encodings[0].ids, vec![1]);
    assert_eq!(encodings[1].ids, vec![2]);
}

#[test]
#[cfg(feature = "backend-hf")]
fn chat_template_renders_generation_prompt() {
    let fixture = hf_fixture();
    let tokenizer = AyakaTokenizer::from_path(fixture.path()).unwrap();
    let prompt = tokenizer
        .apply_chat_template(&[ChatMessage::user("hello")], true)
        .unwrap();

    assert!(prompt.contains("user: hello"));
    assert!(prompt.ends_with("assistant: "));
    assert!(
        !tokenizer
            .encode_chat(&[ChatMessage::user("hello")])
            .unwrap()
            .ids
            .is_empty()
    );
}

#[test]
#[cfg(feature = "backend-hf")]
fn missing_chat_template_is_typed_error() {
    let fixture = hf_fixture();
    std::fs::remove_file(fixture.path().join("tokenizer_config.json")).unwrap();
    let tokenizer = AyakaTokenizer::from_path(fixture.path()).unwrap();
    let err = tokenizer
        .apply_chat_template(&[ChatMessage::user("hello")], true)
        .unwrap_err();

    assert!(matches!(err, TokenizerError::NoChatTemplate));
}

#[test]
fn bad_chat_template_returns_template_error() {
    let err = ChatTemplateRenderer::from_config_value(Some(&json!({
        "chat_template": "{% if"
    })))
    .unwrap_err();

    assert!(matches!(err, TokenizerError::TemplateRenderError(_)));
}

#[test]
#[cfg(feature = "backend-hf")]
fn detokenizer_streams_hf_ascii_and_skips_special_tokens() {
    let fixture = hf_fixture();
    let tokenizer = AyakaTokenizer::from_path(fixture.path()).unwrap();
    let mut detokenizer = tokenizer.create_detokenizer(true);

    assert_eq!(detokenizer.step(3).unwrap(), None);
    assert_eq!(detokenizer.step(1).unwrap(), Some("hello".into()));
    assert_eq!(detokenizer.flush(), None);
}

#[cfg(feature = "backend-tekken")]
#[test]
fn detokenizer_buffers_split_utf8_until_complete() {
    let dir = tempfile::tempdir().unwrap();
    write_json(
        dir.path().join("tekken.json"),
        json!({
            "version": 1,
            "vocab": [
                {"id": 0, "bytes": "4w=="},
                {"id": 1, "bytes": "gQ=="},
                {"id": 2, "bytes": "gg=="},
                {"id": 3, "token": "!"}
            ],
            "special_tokens": {"</s>": 4}
        }),
    );

    let tokenizer = AyakaTokenizer::from_path(dir.path()).unwrap();
    let mut detokenizer = tokenizer.create_detokenizer(true);

    assert_eq!(detokenizer.step(0).unwrap(), None);
    assert_eq!(detokenizer.step(1).unwrap(), None);
    assert_eq!(detokenizer.step(2).unwrap(), Some("あ".into()));
    assert_eq!(detokenizer.step(4).unwrap(), None);
    assert_eq!(detokenizer.step(3).unwrap(), Some("!".into()));

    detokenizer.reset();
    assert_eq!(detokenizer.step(0).unwrap(), None);
    assert_eq!(
        detokenizer.flush(),
        Some(String::from_utf8_lossy(&[0xe3]).into_owned())
    );
}
