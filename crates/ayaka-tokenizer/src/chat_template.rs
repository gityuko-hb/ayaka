use crate::error::{TokenizerError, TokenizerResult};
use minijinja::{Environment, context};
use serde::Serialize;
use serde_json::Value;

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

#[derive(Debug, Clone)]
pub struct ChatTemplateRenderer {
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatTemplateRenderer {
    pub fn from_config_value(config: Option<&Value>) -> TokenizerResult<Option<Self>> {
        let Some(config) = config else {
            return Ok(None);
        };

        let Some(template) = config
            .get("chat_template")
            .and_then(Value::as_str)
        else {
            return Ok(None);
        };

        let renderer = Self {
            template: template.to_owned(),
            bos_token: token_content(config.get("bos_token")),
            eos_token: token_content(config.get("eos_token")),
        };

        renderer.validate()?;
        Ok(Some(renderer))
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> TokenizerResult<String> {
        Environment::new()
            .render_str(
                &self.template,
                context! {
                    messages => messages,
                    add_generation_prompt => add_generation_prompt,
                    bos_token => self.bos_token.as_deref(),
                    eos_token => self.eos_token.as_deref(),
                },
            )
            .map_err(|err| TokenizerError::TemplateRenderError(err.to_string()))
    }

    fn validate(&self) -> TokenizerResult<()> {
        let mut env = Environment::new();
        env.add_template_owned("chat", self.template.clone())
            .map_err(|err| TokenizerError::TemplateRenderError(err.to_string()))
    }
}

fn token_content(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(token) => Some(token.clone()),
        Value::Object(obj) => obj
            .get("content")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    }
}
