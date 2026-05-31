use std::collections::{HashMap, HashSet};

use crate::types::SuffixRepeatPenaltyConfig;
use candle_core::Result;
use tokenizers::Tokenizer;

/// SuffixRepeatPenaltyConfig after stop_tokens has been tokenized.
#[derive(Clone, Debug)]
pub struct SuffixRepeatInner {
    pub stop_tokens: HashSet<u32>,
    pub strength: f32,
    pub growth_rate: f32,
    pub min_match_len: usize,
    pub max_match_len: usize,
    pub position_aware: bool,
}

impl SuffixRepeatInner {
    pub fn from_config(
        config: &SuffixRepeatPenaltyConfig,
        tokenizer: &Tokenizer,
    ) -> anyhow::Result<Self> {
        let breakers = config
            .stop_tokens
            .iter()
            .filter_map(|s| {
                let enc = tokenizer
                    .encode_fast(["a", s.as_str()].concat(), true)
                    .ok()?;
                let ids = enc.get_ids();
                ids.last().copied()
            })
            .collect::<HashSet<_>>();

        Ok(Self {
            stop_tokens: breakers,
            strength: config.strength,
            growth_rate: config.growth_rate,
            min_match_len: config.min_match_len,
            max_match_len: config.max_match_len,
            position_aware: config.position_aware,
        })
    }
}

/// Incremental position index — avoid O(n) scan per sample step.
#[derive(Clone)]
pub struct RepeatPenaltyIndex {
    token_positions: HashMap<u32, Vec<usize>>,
    indexed_len: usize,
    capacity: usize,
}

impl RepeatPenaltyIndex {
    pub fn new(
        context: &[u32],
        capacity: usize,
    ) -> Self {
        let mut idx = Self {
            token_positions: HashMap::new(),
            indexed_len: 0,
            capacity,
        };
        idx.sync(context);
        idx
    }

    pub fn sync(
        &mut self,
        context: &[u32],
    ) {
        if context.len() < self.indexed_len || context.len() > self.capacity {
            self.token_positions.clear();
            self.indexed_len = 0;
        }
        for (pos, &tok) in context[self.indexed_len..].iter().enumerate() {
            self.token_positions
                .entry(tok)
                .or_default()
                .push(self.indexed_len + pos);
        }
        self.indexed_len = context.len();
    }

    pub fn get(
        &self,
        token: u32,
    ) -> Option<&[usize]> {
        self.token_positions
            .get(&token)
            .map(|v| v.as_slice())
    }
}

/// Apply suffix-based repeat penalty
/// O(1) lookup
/// Supports optional position-aware recency weighting.
pub fn apply_suffix_repeat_penalty(
    logits: &mut [f32],
    context: &[u32],
    params: &SuffixRepeatInner,
    index: &mut RepeatPenaltyIndex,
) -> Result<()> {
    if params.strength == 0.0 || context.is_empty() {
        return Ok(());
    }

    index.sync(context);

    let last_token = context[context.len() - 1];
    let n = context.len();

    let Some(matches) = index.get(last_token) else {
        return Ok(());
    };

    struct MatchInfo {
        len: usize,
        nearest_pos: usize,
    }
    let mut match_info: HashMap<u32, MatchInfo> = HashMap::new();

    for &i in matches {
        if i + 1 >= n {
            continue;
        }
        let next_token = context[i + 1];
        if params.stop_tokens.contains(&next_token) {
            continue;
        }

        let mut match_len = 1usize;
        let limit = params.max_match_len.min(i + 1);
        while match_len < limit {
            let prev = context[n - 1 - match_len];
            if context[i - match_len] != prev || params.stop_tokens.contains(&prev) {
                break;
            }
            match_len += 1;
        }

        match_info
            .entry(next_token)
            .and_modify(|m| {
                m.len = m.len.max(match_len);
                m.nearest_pos = m.nearest_pos.max(i);
            })
            .or_insert(MatchInfo {
                len: match_len,
                nearest_pos: i,
            });
    }

    for (&tok, info) in &match_info {
        if info.len < params.min_match_len {
            continue;
        }
        if (tok as usize) >= logits.len() {
            continue;
        }

        let excess = (info.len - params.min_match_len) as f32;
        let base_penalty = params.strength * params.growth_rate.powf(excess);

        let penalty = if params.position_aware {
            let recency = info.nearest_pos as f32 / n as f32;
            base_penalty * (1.0 + recency)
        } else {
            base_penalty
        };

        logits[tok as usize] -= penalty;
    }

    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct RepetitionPenaltyConfig {
    /// Subtract `count × freq_penalty` — penalize proportionally to the number of occurrences.
    pub frequency: f32,

    /// Subtract `presence_penalty` if the token has appeared at least once.
    pub presence: f32,

    /// Divide (if logit > 0) or multiply (if logit < 0) when the token has appeared.
    /// Neutral = 1.0.
    pub repetition: f32,
}

impl RepetitionPenaltyConfig {
    pub fn new(
        frequency: f32,
        presence: f32,
        repetition: f32,
    ) -> Self {
        Self {
            frequency,
            presence,
            repetition,
        }
    }

    pub fn is_neutral(&self) -> bool {
        self.frequency.abs() < f32::EPSILON
            && self.presence.abs() < f32::EPSILON
            && (self.repetition - 1.0).abs() < f32::EPSILON
    }
}

/// Setup to apply freq + presence + repetition penalty
pub fn apply_freq_pres_rep(
    logits: &mut [f32],
    context: &[u32],
    cfg: &RepetitionPenaltyConfig,
) -> Result<()> {
    if cfg.is_neutral() {
        return Ok(());
    }

    let vocab = logits.len();
    let mut counts = vec![0.0f32; vocab];
    for &t in context {
        if (t as usize) < vocab {
            counts[t as usize] += 1.0;
        }
    }

    for (id, logit) in logits.iter_mut().enumerate() {
        let c = counts[id];
        *logit -= c * cfg.frequency;
        if c > 0.0 {
            *logit -= cfg.presence;
            if cfg.repetition != 1.0 {
                if *logit > 0.0 {
                    *logit /= cfg.repetition;
                } else {
                    *logit *= cfg.repetition;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_penalty_reduces_repeated_token() {
        let mut logits = vec![2.0f32, 1.0, 0.5];
        let context = [0u32, 0, 0]; // token 0 appears 3 times 
        let cfg = RepetitionPenaltyConfig::new(0.5, 0.0, 1.0);
        apply_freq_pres_rep(&mut logits, &context, &cfg).unwrap();
        // logit[0] = 2.0 - 3*0.5 = 0.5
        assert!((logits[0] - 0.5).abs() < 1e-5);
        assert!((logits[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_presence_penalty_binary() {
        let mut logits = vec![2.0f32, 1.0];
        let context = [0u32]; // only token 0 appears 
        let cfg = RepetitionPenaltyConfig::new(0.0, 1.0, 1.0);
        apply_freq_pres_rep(&mut logits, &context, &cfg).unwrap();
        assert!((logits[0] - 1.0).abs() < 1e-5); // 2.0 - 1.0 
        assert!((logits[1] - 1.0).abs() < 1e-5); // constant 
    }

    #[test]
    fn test_rep_penalty_divides_positive_logit() {
        let mut logits = vec![4.0f32, 1.0];
        let context = [0u32];
        let cfg = RepetitionPenaltyConfig::new(0.0, 0.0, 2.0);
        apply_freq_pres_rep(&mut logits, &context, &cfg).unwrap();
        assert!((logits[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_neutral_config_is_noop() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let orig = logits.clone();
        let _cfg = RepetitionPenaltyConfig::default(); // repetition=0.0 by default 
        // default repetition is 0.0, need to set to 1.0 for neutral
        let neutral = RepetitionPenaltyConfig::new(0.0, 0.0, 1.0);
        apply_freq_pres_rep(&mut logits, &[0u32], &neutral).unwrap();
        assert_eq!(logits, orig);
    }
}

#[cfg(test)]
mod repeat_penalty_index_tests {
    use super::*;

    #[test]
    fn test_index_lookup() {
        let ctx = vec![10u32, 20, 30, 20, 40];
        let idx = RepeatPenaltyIndex::new(&ctx, 100);
        let pos = idx.get(20).unwrap();
        assert_eq!(pos, &[1, 3]);
        assert!(idx.get(99).is_none());
    }

    #[test]
    fn test_index_incremental() {
        let mut idx = RepeatPenaltyIndex::new(&[0u32, 1, 2], 100);
        assert_eq!(idx.indexed_len, 3);
        idx.sync(&[0u32, 1, 2, 3]);
        assert_eq!(idx.indexed_len, 4);
        let pos = idx.get(3).unwrap();
        assert_eq!(pos, &[3]);
    }

    #[test]
    fn test_index_truncate_rebuild() {
        let mut idx = RepeatPenaltyIndex::new(&[1u32, 2, 3], 5);
        assert_eq!(idx.indexed_len, 3);
        idx.sync(&[4u32, 5]);
        assert_eq!(idx.indexed_len, 2);
        assert!(idx.get(1).is_none());
        assert_eq!(idx.get(4).unwrap(), &[0]);
    }
}

#[cfg(test)]
mod suffix_repeat_tests {
    use super::*;

    fn make_params() -> SuffixRepeatInner {
        SuffixRepeatInner {
            stop_tokens: HashSet::new(),
            strength: 1.0,
            growth_rate: 1.75,
            min_match_len: 2,
            max_match_len: 50,
            position_aware: false,
        }
    }

    #[test]
    fn test_disabled_when_strength_zero() {
        let mut logits = vec![10.0f32, 10.0, 10.0];
        let ctx = vec![0u32, 1, 0, 1, 0];
        let params = SuffixRepeatInner {
            strength: 0.0,
            ..make_params()
        };
        let mut idx = RepeatPenaltyIndex::new(&ctx, 100);
        let orig = logits.clone();
        apply_suffix_repeat_penalty(&mut logits, &ctx, &params, &mut idx).unwrap();
        assert_eq!(logits, orig);
    }

    #[test]
    fn test_penalty_applied_for_repeated_suffix() {
        // ctx [A,B,A,B,A] n=5, last=A, matches at i=0,2
        // i=2: limit=50.min(2)=2, len=1: ctx[1]=B vs ctx[3]=B OK, len=2; 2<2? no → match_len=2
        // next_token=B → excess=0, penalty=1.0*1.75^0=1.0
        let mut logits = vec![10.0f32, 10.0, 10.0];
        let ctx = vec![0u32, 1, 0, 1, 0];
        let params = make_params();
        let mut idx = RepeatPenaltyIndex::new(&ctx, 100);
        apply_suffix_repeat_penalty(&mut logits, &ctx, &params, &mut idx).unwrap();
        assert!((logits[0] - 10.0).abs() < 1e-5);
        assert!((logits[1] - 8.25).abs() < 1e-5);
    }

    #[test]
    fn test_min_match_len_threshold() {
        // ctx [0,1,0,2,0] n=5, last=0, matches at i=0,2
        // i=2: context[1]=1 vs context[3]=2 → 1!=2 → break → match_len=1
        // 1 < min_match_len=2 → no penalty
        let mut logits = vec![10.0f32, 10.0];
        let ctx = vec![0u32, 1, 0, 2, 0];
        let params = make_params();
        let mut idx = RepeatPenaltyIndex::new(&ctx, 100);
        let orig = logits.clone();
        apply_suffix_repeat_penalty(&mut logits, &ctx, &params, &mut idx).unwrap();
        assert_eq!(logits, orig);
    }

    #[test]
    fn test_max_match_len_cap() {
        // repeated [0,1] pattern with max_match_len=3
        let mut logits = vec![10.0f32, 10.0];
        let ctx: Vec<u32> = (0..11u32).map(|i| i % 2).collect();
        let params = SuffixRepeatInner {
            max_match_len: 3,
            ..make_params()
        };
        let mut idx = RepeatPenaltyIndex::new(&ctx, 100);
        apply_suffix_repeat_penalty(&mut logits, &ctx, &params, &mut idx).unwrap();
        // excess = 3-2 = 1, penalty = 1.0*1.75^1 = 1.75
        // logit[1] = 10.0 - 1.75 = 8.25
        assert!((logits[1] - 8.25).abs() < 1e-5);
    }

    #[test]
    fn test_stop_tokens_break_match() {
        let mut logits = vec![10.0f32, 10.0, 10.0];
        let ctx = vec![0u32, 1, 0, 2, 0, 1, 0];
        // last=0, i=4: next=1, limit=50.min(4)=4
        // len=1: context[3]=2 vs context[5]=1 → 2!=1 → break → match_len=1
        // With stop_token at 2, i=2: next=2 is stop → skipped
        let params = SuffixRepeatInner {
            stop_tokens: [2u32].into_iter().collect(),
            ..make_params()
        };
        let mut idx = RepeatPenaltyIndex::new(&ctx, 100);
        let orig = logits.clone();
        apply_suffix_repeat_penalty(&mut logits, &ctx, &params, &mut idx).unwrap();
        assert_eq!(logits, orig);
    }

    #[test]
    fn test_position_aware_recency() {
        // ctx: [0,1,2,0,1,2,0,1,2] n=9, last=2, matches at i=2,5
        // i=5: next=0, limit=50.min(5)=5, match_len extends to 5
        // max match_len=5, nearest_pos=5
        // Without position_aware: excess=3, penalty=1.0*1.75^3=5.359
        // With position_aware: recency=5/9=0.556, penalty=5.359*1.556=8.337
        let ctx = vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2];
        let mut logits_no = vec![10.0f32; 3];
        let mut logits_yes = vec![10.0f32; 3];

        let params_no = SuffixRepeatInner {
            position_aware: false,
            ..make_params()
        };
        let mut idx_no = RepeatPenaltyIndex::new(&ctx, 100);
        apply_suffix_repeat_penalty(&mut logits_no, &ctx, &params_no, &mut idx_no).unwrap();

        let params_yes = SuffixRepeatInner {
            position_aware: true,
            ..make_params()
        };
        let mut idx_yes = RepeatPenaltyIndex::new(&ctx, 100);
        apply_suffix_repeat_penalty(&mut logits_yes, &ctx, &params_yes, &mut idx_yes).unwrap();

        let penalty_no = 10.0 - logits_no[0];
        let penalty_yes = 10.0 - logits_yes[0];
        assert!(
            penalty_yes > penalty_no,
            "position_aware penalty ({penalty_yes}) should exceed non-aware ({penalty_no})"
        );
    }
}
