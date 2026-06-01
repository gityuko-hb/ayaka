use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VocabMetadata {
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub special_token_ids: HashSet<u32>,
}

impl VocabMetadata {
    pub fn from_config_value(
        config: Option<&Value>,
        vocab_size: usize,
    ) -> Self {
        let mut metadata = Self {
            vocab_size,
            ..Self::default()
        };

        let Some(config) = config else {
            return metadata;
        };

        metadata.bos_token_id = token_id(config.get("bos_token_id"));
        metadata.eos_token_id = token_id(config.get("eos_token_id"));
        metadata.pad_token_id = token_id(config.get("pad_token_id"));
        metadata.unk_token_id = token_id(config.get("unk_token_id"));

        for id in [
            metadata.bos_token_id,
            metadata.eos_token_id,
            metadata.pad_token_id,
            metadata.unk_token_id,
        ]
        .into_iter()
        .flatten()
        {
            metadata.special_token_ids.insert(id);
        }

        collect_ids(
            config.get("all_special_ids"),
            &mut metadata.special_token_ids,
        );
        collect_ids(
            config.get("additional_special_tokens_ids"),
            &mut metadata.special_token_ids,
        );
        collect_added_token_ids(
            config.get("added_tokens_decoder"),
            &mut metadata.special_token_ids,
        );

        metadata
    }

    pub fn merge_backend_special_ids(
        &mut self,
        ids: impl IntoIterator<Item = u32>,
    ) {
        self.special_token_ids.extend(ids);
    }
}

pub(crate) fn token_id(value: Option<&Value>) -> Option<u32> {
    match value? {
        Value::Number(n) => n.as_u64().and_then(|id| u32::try_from(id).ok()),
        Value::Object(obj) => obj
            .get("id")
            .and_then(|id| id.as_u64())
            .and_then(|id| u32::try_from(id).ok()),
        _ => None,
    }
}

fn collect_ids(
    value: Option<&Value>,
    out: &mut HashSet<u32>,
) {
    let Some(Value::Array(values)) = value else {
        return;
    };

    out.extend(
        values
            .iter()
            .filter_map(|value| token_id(Some(value))),
    );
}

fn collect_added_token_ids(
    value: Option<&Value>,
    out: &mut HashSet<u32>,
) {
    let Some(Value::Object(tokens)) = value else {
        return;
    };

    for (key, value) in tokens {
        if value
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && let Ok(id) = key.parse::<u32>() {
                out.insert(id);
             }
    }
}
