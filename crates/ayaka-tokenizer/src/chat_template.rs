use crate::error::{TokenizerError, TokenizerResult};
use minijinja::{Environment, Value, context};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant { content: String },
    #[serde(rename = "tool")]
    Tool {
        content: String,
        tool_call_id: String,
    },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
        }
    }
}

/// Pre-compiled Jinja2 chat template renderer.
///
/// The minijinja `Environment` is allocated behind `Arc` so that:
/// - `Clone` is O(1) (refcount increment only).
/// - The renderer is `Send + Sync` and safe to share across request threads.
#[derive(Debug, Clone)]
pub struct ChatTemplateRenderer {
    env: Arc<Environment<'static>>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplateRenderer {
    pub fn from_config_value(config: Option<&JsonValue>) -> TokenizerResult<Option<Self>> {
        let Some(config) = config else {
            return Ok(None);
        };

        let Some(template) = config
            .get("chat_template")
            .and_then(JsonValue::as_str)
        else {
            return Ok(None);
        };

        let mut env = Environment::new();

        // register `raise_exception` — used by Llama 3, Mistral, Phi-3,
        // and other chat templates to enforce correct conversation structure.
        // Without this, templates that hit an error branch silently produce wrong
        // output instead of surfacing the problem.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        // pre-compile the template string once at construction time.
        // The original code called `Environment::new().render_str(...)` on
        // every render(), re-parsing the template on every request. Now the
        // template is parsed and stored once; render() just retrieves it.
        env.add_template_owned("chat", template.to_owned())
            .map_err(|err| TokenizerError::TemplateRenderError(err.to_string()))?;

        Ok(Some(Self {
            env: Arc::new(env),
            bos_token: token_content(config.get("bos_token")),
            eos_token: token_content(config.get("eos_token")),
        }))
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> TokenizerResult<String> {
        // pre-compile the template string once at construction time.
        // The original code called `Environment::new().render_str(...)` on
        // every render(), re-parsing the template on every request. Now the
        // template is parsed and stored once; render() just retrieves it.
        self.env
            .get_template("chat")
            .map_err(|err| TokenizerError::TemplateRenderError(err.to_string()))?
            .render(context! {
                messages,
                add_generation_prompt,
                bos_token => self.bos_token.as_deref(),
                eos_token => self.eos_token.as_deref(),
            })
            .map_err(|err| TokenizerError::TemplateRenderError(err.to_string()))
    }
}

fn token_content(value: Option<&JsonValue>) -> Option<String> {
    match value? {
        JsonValue::String(token) => Some(token.clone()),
        JsonValue::Object(obj) => obj
            .get("content")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    }
}
