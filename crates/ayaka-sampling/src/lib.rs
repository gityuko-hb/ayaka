//! `ayaka-sampling` — modular sampling pipeline + algorithm for LLM inference.

pub mod grammar;
pub mod logits;
pub mod min_p;
pub mod processor;
pub mod random;
pub mod repetition_penalty;
pub mod temperature;
pub mod top_k;
pub mod top_p;
pub mod types;

// Re-exports cho convenience
pub use grammar::{AllowListGrammar, GrammarMask, NoGrammar};
pub use logits::LogitsProcessor;
pub use random::RngStrategy;
pub use types::{
    Logprobs, ModelGenerationDefaults, SamplingParams, StopTokens, SuffixRepeatPenaltyConfig,
    TopLogprob,
};

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use tokenizers::Tokenizer;

use grammar::{GrammarCheck, check_token};
use min_p::apply_min_p;
use processor::ProcessorChain;
use random::{build_logprobs, normalize, sample_from_filtered};
use repetition_penalty::{RepeatPenaltyIndex, RepetitionPenaltyConfig, SuffixRepeatInner};
use std::cell::RefCell;
use temperature::{TemperatureOutput, apply_temperature};
use top_k::apply_top_k;
use top_p::apply_top_p;

// SamplerBuilder - Main State
/// Builder with combo validated - Avoid invalid states.
#[derive(Default)]
pub struct SamplerBuilder {
    temperature: Option<f64>,
    top_k: usize,
    top_p: f64,
    min_p: f64,
    top_n_logprobs: usize,
    rep_config: RepetitionPenaltyConfig,
    suffix_repeat: Option<SuffixRepeatPenaltyConfig>,
    max_context_len: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    processors: Vec<Arc<dyn LogitsProcessor>>,
    /// Token-level additive bias applied after processor chain, before temperature.
    /// Positive values boost a token; negative values suppress it; NEG_INFINITY bans it.
    logits_bias: Option<std::collections::HashMap<u32, f32>>,
}

impl SamplerBuilder {
    pub fn temperature(
        mut self,
        t: f64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(t >= 0.0, "temperature must be >= 0, accepted {t}");
        self.temperature = if t < 1e-7 {
            None
        } else {
            Some(t)
        };
        Ok(self)
    }

    pub fn top_k(
        mut self,
        k: usize,
    ) -> Self {
        self.top_k = k;
        self
    }

    pub fn top_p(
        mut self,
        p: f64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (0.0..=1.0).contains(&p),
            "top_p must be in [0,1], accepted {p}"
        );
        self.top_p = p;
        Ok(self)
    }

    pub fn min_p(
        mut self,
        p: f64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (0.0..=1.0).contains(&p),
            "min_p must be in [0,1], accepted {p}"
        );
        self.min_p = p;
        Ok(self)
    }

    pub fn top_n_logprobs(
        mut self,
        n: usize,
    ) -> Self {
        self.top_n_logprobs = n;
        self
    }

    pub fn repetition_penalty(
        mut self,
        frequency: f32,
        presence: f32,
        repetition: f32,
    ) -> Self {
        self.rep_config = RepetitionPenaltyConfig::new(frequency, presence, repetition);
        self
    }

    pub fn suffix_repeat(
        mut self,
        params: SuffixRepeatPenaltyConfig,
    ) -> Self {
        self.suffix_repeat = Some(params);
        self
    }

    /// Set maximum context length for the position index.
    /// When the context exceeds this, the index is rebuilt.
    /// `0` (default) means unlimited — index grows without rebuilding.
    pub fn max_context_len(
        mut self,
        n: usize,
    ) -> Self {
        self.max_context_len = n;
        self
    }

    pub fn tokenizer(
        mut self,
        tok: Arc<Tokenizer>,
    ) -> Self {
        self.tokenizer = Some(tok);
        self
    }

    pub fn add_processor(
        mut self,
        p: Arc<dyn LogitsProcessor>,
    ) -> Self {
        self.processors.push(p);
        self
    }

    /// Set per-token logits bias applied after the processor chain, before temperature scaling.
    /// Positive values boost a token; negative values suppress it; NEG_INFINITY bans it.
    /// Replaces any previously set bias map entirely.
    pub fn logits_bias(
        mut self,
        bias: std::collections::HashMap<u32, f32>,
    ) -> Self {
        self.logits_bias = if bias.is_empty() {
            None
        } else {
            Some(bias)
        };
        self
    }

    pub fn build(
        self,
        initial_context: &[u32],
    ) -> anyhow::Result<Sampler> {
        let suffix_repeat_inner = match (self.suffix_repeat, &self.tokenizer) {
            (Some(p), Some(tok)) => Some(SuffixRepeatInner::from_config(&p, tok)?),
            (Some(_), None) => {
                anyhow::bail!("SuffixRepeatPenaltyConfig requires a tokenizer");
            },
            (None, _) => None,
        };

        let capacity = if self.max_context_len == 0 {
            usize::MAX
        } else {
            self.max_context_len.max(initial_context.len())
        };
        let index = RepeatPenaltyIndex::new(initial_context, capacity);

        Ok(Sampler {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p as f32,
            min_p: self.min_p as f32,
            top_n_logprobs: self.top_n_logprobs,
            rep_config: self.rep_config,
            suffix_repeat_inner,
            suffix_repeat_index: RefCell::new(index),
            tokenizer: self.tokenizer,
            chain: ProcessorChain::new(self.processors),
            logits_bias: self.logits_bias,
        })
    }
}

/// The sampler synthesizes the entire pipeline.
/// # Pipeline
///
/// ```text
/// logits (raw, Tensor)
/// → apply_suffix_repeat_penalty (repetition_penalty)   [O(1) incremental index]
/// → apply_freq_pres_rep         (repetition_penalty)
/// → ProcessorChain              (processor)            [custom transforms]
/// → apply_logits_bias           (processor)            [wired from SamplingParams]
/// → apply_temperature + softmax (temperature)
/// → apply_top_k                 (top_k)                [partial sort O(n)+O(k log k)]
/// → apply_top_p                 (top_p)                [nucleus, uses post-top-k order]
/// → rebuild idx_probs                                  [max_p for min-p is post-top-p]
/// → apply_min_p                 (min_p)                [threshold relative to post-top-p max]
/// → sample_from_filtered / argmax (random)             [WeightedIndex on k tokens only]
/// → grammar check / resample    (grammar)              [GrammarMask trait, decoupled]
/// → build_logprobs              (random)               [ln, OpenAI-compatible]
/// ```
///
/// # Thread safety
/// `Sampler` is `!Sync` because `suffix_repeat_index` uses `RefCell`.
/// Use one `Sampler` per sequence; clone for each new sequence.
#[derive(Clone)]
pub struct Sampler {
    temperature: Option<f64>,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    top_n_logprobs: usize,
    rep_config: RepetitionPenaltyConfig,
    suffix_repeat_inner: Option<SuffixRepeatInner>,
    /// Interior-mutable index — Sampler is intentionally !Sync.
    /// Use one Sampler per sequence; do not share across threads.
    suffix_repeat_index: RefCell<RepeatPenaltyIndex>,
    tokenizer: Option<Arc<Tokenizer>>,
    chain: ProcessorChain,
    /// Per-token additive logits bias, wired from SamplingParams::logits_bias.
    /// Applied after the processor chain, before temperature scaling.
    logits_bias: Option<std::collections::HashMap<u32, f32>>,
}

impl Sampler {
    pub fn builder() -> SamplerBuilder {
        SamplerBuilder::default()
    }

    /// Quickly create patterns using `SamplingParams` — use when you don't need a builder pattern.
    pub fn from_params(
        params: &SamplingParams,
        tokenizer: Option<Arc<Tokenizer>>,
        initial_context: &[u32],
    ) -> anyhow::Result<Self> {
        let mut b = Self::builder()
            .top_n_logprobs(params.top_n_logprobs)
            .top_k(params.top_k.unwrap_or(0))
            .repetition_penalty(
                params.frequency_penalty.unwrap_or(0.0),
                params.presence_penalty.unwrap_or(0.0),
                params.repetition_penalty.unwrap_or(1.0),
            );

        if let Some(t) = params.temperature {
            b = b.temperature(t)?;
        }
        if let Some(p) = params.top_p {
            b = b.top_p(p)?;
        }
        if let Some(mp) = params.min_p {
            b = b.min_p(mp)?;
        }
        if let Some(sp) = &params.suffix_repeat {
            b = b.suffix_repeat(sp.clone());
        }
        if let Some(tok) = tokenizer {
            b = b.tokenizer(tok);
        }
        if let Some(ref bias) = params.logits_bias
            && !bias.is_empty()
        {
            b = b.logits_bias(bias.clone());
        }
        b.build(initial_context)
    }

    /// Argmax mode with temperature = None.
    pub fn is_argmax(&self) -> bool {
        self.temperature.is_none()
    }

    fn apply_all_penalties(
        &self,
        mut logits_vec: Vec<f32>,
        context: &[u32],
    ) -> Result<Vec<f32>> {
        if let Some(ref sp) = self.suffix_repeat_inner {
            let mut index = self.suffix_repeat_index.borrow_mut();
            repetition_penalty::apply_suffix_repeat_penalty(
                &mut logits_vec,
                context,
                sp,
                &mut index,
            )?;
        }
        repetition_penalty::apply_freq_pres_rep(&mut logits_vec, context, &self.rep_config)?;
        Ok(logits_vec)
    }

    fn probs_from_logits(
        &self,
        logits: Tensor,
    ) -> Result<(Vec<f32>, bool)> {
        match apply_temperature(logits, self.temperature)? {
            TemperatureOutput::Argmax(one_hot) => Ok((one_hot, true)),
            TemperatureOutput::Probs(probs) => Ok((probs, false)),
        }
    }

    fn filter_probs(
        &self,
        probs: &mut [f32],
    ) -> Vec<(u32, f32)> {
        // Step 1: top-k — zero out tokens outside top-k, return sorted idx_probs.
        let idx_probs_after_topk = apply_top_k(probs, self.top_k);

        // Step 2: top-p — zero out tail tokens in probs[] using top-k order.
        apply_top_p(probs, &idx_probs_after_topk, self.top_p);

        // Step 3: rebuild idx_probs to reflect what top-p actually kept,
        // so min_p computes max_p from the *post-top-p* distribution, not pre-top-p.
        // Without this, if top-p zeroed out the highest-prob token (edge case),
        // min_p threshold would be anchored to a token that's no longer in the nucleus.
        let idx_probs_after_topp: Vec<(u32, f32)> = idx_probs_after_topk
            .iter()
            .filter(|(idx, _)| probs[*idx as usize] > 0.0)
            .map(|&(idx, _)| (idx, probs[idx as usize]))
            .collect();

        // Step 4: min-p — uses post-top-p idx_probs, so max_p is accurate.
        apply_min_p(probs, &idx_probs_after_topp, self.min_p);

        // Step 5: collect final survivors.
        idx_probs_after_topp
            .into_iter()
            .filter(|(idx, _)| probs[*idx as usize] > 0.0)
            .map(|(idx, _)| (idx, probs[idx as usize]))
            .collect()
    }

    /// Sample a token from the logits tensor.
    ///
    /// `grammar` — pass `&NoGrammar` if there are no constraints.
    pub fn sample<G: GrammarMask>(
        &self,
        logits: Tensor,
        context: &[u32],
        return_logprobs: bool,
        rng: &RngStrategy,
        grammar: &mut G,
    ) -> Result<Logprobs> {
        let device = logits.device().clone();
        let logits = logits.to_dtype(DType::F32)?;

        // 1. Penalties
        let logits_vec = logits.to_vec1::<f32>()?;
        let logits_vec = self.apply_all_penalties(logits_vec, context)?;
        let logits = Tensor::from_vec(logits_vec, logits.shape().dims1()?, &device)?;

        // 2. Custom processors
        let logits = self.chain.apply(logits, context)?;

        // 3. Logits bias — applied after processor chain, before temperature.
        let logits = if let Some(ref bias) = self.logits_bias {
            processor::apply_logits_bias(&logits, bias, &device)?
        } else {
            logits
        };

        // 4. Temperature + softmax → probs
        let (mut probs, is_argmax) = self.probs_from_logits(logits.clone())?;
        let all_probs = probs.clone();

        // 5. Filtering (top_k → top_p → min_p)
        let filtered = if is_argmax {
            // Argmax: just need token with prob = 1.0
            probs
                .iter()
                .enumerate()
                .find(|&(_, p)| *p > 0.0)
                .map(|(i, &p)| vec![(i as u32, p)])
                .unwrap_or_default()
        } else {
            self.filter_probs(&mut probs)
        };

        // 6. Sample or argmax
        let (token, prob) = if filtered.is_empty() {
            // fallback — Get argmax from all_probs
            let best = logits::argmax_f32(&all_probs);
            (best, all_probs[best as usize])
        } else if is_argmax {
            filtered[0]
        } else {
            sample_from_filtered(&filtered, rng)?
        };

        // 7. Grammar check — Resample if needed.
        let (final_token, final_prob) = match check_token(grammar, token, all_probs.len(), &device)?
        {
            GrammarCheck::Allowed => (token, prob),
            GrammarCheck::Disallowed(bias) => {
                // Resample with grammar bias.
                // `logits` here is the tensor after penalties + processor chain + logits_bias,
                // but *before* temperature — exactly the right point to add the grammar mask.
                // filter_probs runs fresh on the biased distribution, which may admit tokens
                // outside the original top-k; this is intentional — the grammar overrides
                // sampling constraints to ensure a valid token is always returned.
                let biased = grammar::apply_grammar_bias(&logits, &bias)?;
                let (mut biased_probs, biased_is_argmax) = self.probs_from_logits(biased)?;
                let biased_filtered = if biased_is_argmax {
                    biased_probs
                        .iter()
                        .enumerate()
                        .find(|&(_, p)| *p > 0.0)
                        .map(|(i, &p)| vec![(i as u32, p)])
                        .unwrap_or_default()
                } else {
                    self.filter_probs(&mut biased_probs)
                };
                if biased_filtered.is_empty() {
                    (token, prob) // Grammar cannot be satisfied — fall back to original token
                } else if biased_is_argmax {
                    biased_filtered[0]
                } else {
                    sample_from_filtered(&biased_filtered, rng)?
                }
            },
        };

        // 8. Consume grammar token
        if !grammar.is_stopped() {
            grammar.consume(final_token)?;
        }

        // 9. Build Logprobs
        build_logprobs(
            final_token,
            final_prob,
            &all_probs,
            if return_logprobs {
                self.top_n_logprobs
            } else {
                0
            },
            self.tokenizer.as_deref(),
        )
    }

    /// Calculate full probability distribution — used for speculative decoding.
    pub fn speculative_probs(
        &self,
        logits: Tensor,
        context: &[u32],
    ) -> Result<Vec<f32>> {
        let device = logits.device().clone();
        let logits = logits.to_dtype(DType::F32)?;

        let logits_vec = logits.to_vec1::<f32>()?;
        let logits_vec = self.apply_all_penalties(logits_vec, context)?;
        let logits = Tensor::from_vec(logits_vec, logits.shape().dims1()?, &device)?;
        let logits = self.chain.apply(logits, context)?;

        // Apply logits bias before temperature (same order as sample()).
        let logits = if let Some(ref bias) = self.logits_bias {
            processor::apply_logits_bias(&logits, bias, &device)?
        } else {
            logits
        };

        let (mut probs, _) = self.probs_from_logits(logits)?;
        // Use the same corrected filter order as filter_probs():
        // top-k → top-p → rebuild idx_probs → min-p.
        let idx_probs_k = apply_top_k(&mut probs, self.top_k);
        apply_top_p(&mut probs, &idx_probs_k, self.top_p);
        let idx_probs_p: Vec<(u32, f32)> = idx_probs_k
            .iter()
            .filter(|(idx, _)| probs[*idx as usize] > 0.0)
            .map(|&(idx, _)| (idx, probs[idx as usize]))
            .collect();
        apply_min_p(&mut probs, &idx_probs_p, self.min_p);
        normalize(&mut probs)?;
        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    fn make_sampler_argmax() -> Sampler {
        Sampler::builder().build(&[0u32]).unwrap()
    }

    fn make_sampler_temp(t: f64) -> Sampler {
        Sampler::builder()
            .temperature(t)
            .unwrap()
            .build(&[0u32])
            .unwrap()
    }

    #[test]
    fn test_argmax_picks_max() {
        let s = make_sampler_argmax();
        let logits = Tensor::arange(0f32, 8f32, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        let lp = s
            .sample(
                logits,
                &(0..8u32).collect::<Vec<_>>(),
                false,
                &rng,
                &mut NoGrammar,
            )
            .unwrap();
        assert_eq!(lp.token, 7);
    }

    #[test]
    fn test_temperature_logprob_is_ln() {
        let s = make_sampler_temp(1.0);
        let logits = Tensor::from_vec(vec![0.0f32, 100.0, 0.0], 3, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        let lp = s
            .sample(logits, &[0u32], false, &rng, &mut NoGrammar)
            .unwrap();
        assert_eq!(lp.token, 1);
        // prob ≈ 1.0, ln(1.0) ≈ 0.0
        assert!(lp.logprob.abs() < 0.01, "logprob={}", lp.logprob);
    }

    #[test]
    fn test_grammar_constraint_forces_allowed_token() {
        let s = make_sampler_argmax();
        // Logits: Token 2 has the highest logit value, but the grammar only allows token 0.
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 10.0], 3, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        let mut grammar = AllowListGrammar {
            allowed: [0u32].into_iter().collect(),
            stopped: false,
        };
        let lp = s
            .sample(logits, &[0u32], false, &rng, &mut grammar)
            .unwrap();
        assert_eq!(lp.token, 0);
    }

    #[test]
    fn test_sampling_params_roundtrip() {
        let mut params = SamplingParams::neutral();
        params.apply_model_defaults(&ModelGenerationDefaults {
            do_sample: Some(true),
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.05),
            repetition_penalty: Some(1.1),
            max_new_tokens: Some(512),
            max_length: None,
        });
        assert_eq!(params.temperature, Some(0.8));
        assert_eq!(params.top_k, Some(40));
        assert_eq!(params.max_len, Some(512));
    }

    #[test]
    fn test_sampling_params_suffix_repeat() {
        let params = SamplingParams {
            suffix_repeat: Some(SuffixRepeatPenaltyConfig::new(
                1.0, None, None, None, None, None,
            )),
            ..SamplingParams::neutral()
        };
        assert!(params.suffix_repeat.is_some());
    }

    // Setup: 4 tokens with probs [0.6, 0.25, 0.1, 0.05], top_p=0.85.
    // After top-p: cumsum reaches 0.6+0.25=0.85 → token 2 (0.1) is zeroed out.
    // min_p=0.15 with max_p=0.6 → threshold=0.09.
    //   Old code: max_p from pre-top-p idx_probs = 0.6, threshold=0.09 → keeps token 2 (0.1 > 0.09)
    //   New code: max_p from post-top-p = 0.6 (same here, but idx_probs[2] is already zero)
    //             → token 2 correctly absent from filtered list.
    // This test verifies token 2 is NOT in the final filtered set.
    #[test]
    fn test_filter_probs_min_p_uses_post_topp_distribution() {
        let s = Sampler::builder()
            .temperature(1.0)
            .unwrap()
            .top_k(0)   // no top-k limit
            .top_p(0.85)
            .unwrap()
            .min_p(0.15)
            .unwrap()
            .build(&[0u32])
            .unwrap();

        // Provide logits that produce probs ≈ [0.6, 0.25, 0.1, 0.05] after softmax.
        // Use a high-temperature trick: logits proportional to desired probs (unnormalized).
        // ln(0.6)≈-0.51, ln(0.25)≈-1.39, ln(0.1)≈-2.30, ln(0.05)≈-2.996 — but easier
        // to just use logits=[2.77, 1.61, 0.41, -0.30] which softmax to ≈above.
        let logits = Tensor::from_vec(vec![2.77f32, 1.61, 0.41, -0.30], 4, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        // Run 100 times — token 2 and token 3 must never be selected.
        for _ in 0..100 {
            let lp = s
                .sample(logits.clone(), &[0u32], false, &rng, &mut NoGrammar)
                .unwrap();
            assert!(
                lp.token == 0 || lp.token == 1,
                "token {} should have been filtered out",
                lp.token
            );
        }
    }

    // Token 0 has the highest logit (argmax), but we apply bias=-100 to it.
    // Result: token 1 should win.
    #[test]
    fn test_logits_bias_suppresses_token() {
        let mut bias = HashMap::new();
        bias.insert(0u32, -100.0f32);

        let s = Sampler::builder()
            .logits_bias(bias)
            .build(&[0u32])
            .unwrap();

        let logits = Tensor::from_vec(vec![10.0f32, 1.0, 0.5], 3, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        let lp = s
            .sample(logits, &[0u32], false, &rng, &mut NoGrammar)
            .unwrap();
        assert_eq!(
            lp.token, 1,
            "token 0 should be suppressed by logits_bias=-100"
        );
    }

    // logits_bias from SamplingParams roundtrip
    #[test]
    fn test_logits_bias_from_params() {
        let mut bias = HashMap::new();
        bias.insert(2u32, -100.0f32);
        let params = SamplingParams {
            logits_bias: Some(bias),
            ..SamplingParams::neutral()
        };
        let sampler = Sampler::from_params(&params, None, &[0u32]).unwrap();
        // Bias is wired: logits_bias field should be Some
        assert!(sampler.logits_bias.is_some());
        assert_eq!(
            sampler.logits_bias.as_ref().unwrap().get(&2u32),
            Some(&-100.0f32)
        );
    }

    // grammar resample respects argmax path
    #[test]
    fn test_grammar_resample_argmax_path() {
        let s = make_sampler_argmax();
        // Highest logit = token 2, but only token 0 allowed.
        let logits = Tensor::from_vec(vec![1.0f32, 0.5, 10.0], 3, &Device::Cpu).unwrap();
        let rng = RngStrategy::ThreadLocal;
        let mut grammar = AllowListGrammar {
            allowed: [0u32].into_iter().collect(),
            stopped: false,
        };
        // Run multiple times to confirm determinism.
        for _ in 0..5 {
            let lp = s
                .sample(logits.clone(), &[0u32], false, &rng, &mut grammar)
                .unwrap();
            assert_eq!(lp.token, 0);
        }
    }
}
