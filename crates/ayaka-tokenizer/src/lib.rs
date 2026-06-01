//! Tokenization boundary for Ayaka inference.
//!
//! `AyakaTokenizer` loads a model tokenizer once, exposes backend-neutral
//! encode/decode APIs, and creates per-request `StreamingDetokenizer` state for
//! token-by-token generation.
//!
//! ```no_run
//! use ayaka_tokenizer::{AyakaTokenizer, ChatMessage};
//! use std::path::Path;
//!
//! # fn run() -> ayaka_tokenizer::TokenizerResult<()> {
//! let tokenizer = AyakaTokenizer::from_path(Path::new("/models/llama3"))?;
//! let input = tokenizer.encode_chat(&[ChatMessage::user("Hello")])?;
//! let mut stream = tokenizer.create_detokenizer(true);
//! for token_id in input.ids {
//!     let _ = stream.step(token_id)?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod backend;
pub mod chat_template;
pub mod detokenizer;
pub mod error;
pub mod tokenizer;
pub mod tokens;
pub mod vocab;

pub use chat_template::{ChatMessage, ChatTemplateRenderer};
pub use detokenizer::StreamingDetokenizer;
pub use error::{TokenizerError, TokenizerResult};
pub use tokenizer::{AyakaTokenizer, Encoding};
pub use tokens::{TokenSource, get_token};
pub use vocab::VocabMetadata;
