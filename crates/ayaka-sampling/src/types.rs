use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopLogprob {
    pub token: u32,
    /// Natural log (ln), compatible with OpenAI spec..
    pub logprob: f32,
    pub bytes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StopTokens {
    Seqs(Vec<String>),
    Ids(Vec<u32>),
}

static DEFAULT_STOP_TOKENS: std::sync::LazyLock<Vec<String>> =
    std::sync::LazyLock::new(|| ["\n", ":", "\"", "*"].map(String::from).to_vec());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuffixRepeatPenaltyConfig {
    pub strength: f32,
    pub growth_rate: f32,
    pub min_match_len: usize,
    pub max_match_len: usize,
    pub stop_tokens: Vec<String>,
    pub position_aware: bool,
}

impl SuffixRepeatPenaltyConfig {
    pub fn new(
        strength: f32,
        growth_rate: Option<f32>,
        min_match_len: Option<usize>,
        max_match_len: Option<usize>,
        stop_tokens: Option<Vec<String>>,
        position_aware: Option<bool>,
    ) -> Self {
        Self {
            strength,
            growth_rate: growth_rate.unwrap_or(1.75),
            min_match_len: min_match_len.unwrap_or(2),
            max_match_len: max_match_len.unwrap_or(50),
            stop_tokens: stop_tokens.unwrap_or_else(|| DEFAULT_STOP_TOKENS.clone()),
            position_aware: position_aware.unwrap_or(false),
        }
    }
}

impl Default for SuffixRepeatPenaltyConfig {
    fn default() -> Self {
        Self {
            strength: 0.0,
            growth_rate: 1.75,
            min_match_len: 2,
            max_match_len: 50,
            stop_tokens: DEFAULT_STOP_TOKENS.clone(),
            position_aware: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelGenerationDefaults {
    pub do_sample: Option<bool>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub max_new_tokens: Option<usize>,
    pub max_length: Option<usize>,
}

impl ModelGenerationDefaults {
    pub fn is_empty(&self) -> bool {
        self.do_sample.is_none()
            && self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
            && self.max_new_tokens.is_none()
            && self.max_length.is_none()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub top_n_logprobs: usize,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_toks: Option<StopTokens>,
    pub max_len: Option<usize>,
    pub logits_bias: Option<HashMap<u32, f32>>,
    pub n_choices: usize,
    pub suffix_repeat: Option<SuffixRepeatPenaltyConfig>,
}

impl SamplingParams {
    /// Neutral baseline.
    pub fn neutral() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            top_n_logprobs: 0,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_toks: None,
            max_len: None,
            logits_bias: None,
            n_choices: 1,
            suffix_repeat: None,
        }
    }

    /// Greedy — hard argmax.
    pub fn deterministic() -> Self {
        Self {
            top_k: Some(1),
            ..Self::neutral()
        }
    }

    /// Merge model-level defaults into request params.
    pub fn apply_model_defaults(
        &mut self,
        d: &ModelGenerationDefaults,
    ) {
        if d.do_sample == Some(false) {
            self.temperature = None;
            self.top_k = Some(1);
            self.top_p = None;
            self.min_p = None;
        }
        if let Some(t) = d.temperature {
            self.temperature = if t == 0.0 {
                None
            } else {
                Some(t)
            };
        }
        if let Some(k) = d.top_k {
            self.top_k = if k == 0 {
                None
            } else {
                Some(k)
            };
        }
        if let Some(p) = d.top_p {
            self.top_p = Some(p);
        }
        if let Some(mp) = d.min_p {
            self.min_p = Some(mp);
        }
        if let Some(rp) = d.repetition_penalty {
            self.repetition_penalty = Some(rp);
        }
        if let Some(mn) = d.max_new_tokens {
            self.max_len = Some(mn);
        }
    }
}
