use std::sync::{Arc, Mutex};

use candle_core::{Error, Result};
use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand_chacha::ChaCha20Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::{Logprobs, TopLogprob};

/// RNG Strategy
/// Abstraction layer for RNG — allows strategy swapping without changing the call-site.
pub enum RngStrategy {
    ThreadLocal,
    Shared(Arc<Mutex<ChaCha20Rng>>),
}

impl RngStrategy {
    pub fn thread_local_seeded(seed: u64) -> Self {
        THREAD_RNG.with(|cell| {
            *cell.borrow_mut() = Xoshiro256PlusPlus::seed_from_u64(seed);
        });
        Self::ThreadLocal
    }

    pub fn shared(rng: Arc<Mutex<ChaCha20Rng>>) -> Self {
        Self::Shared(rng)
    }
}

thread_local! {
    static THREAD_RNG: std::cell::RefCell<Xoshiro256PlusPlus> = std::cell::RefCell::new(
        Xoshiro256PlusPlus::seed_from_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64,
        )
    )
}

pub fn sample_from_filtered(
    idx_probs: &[(u32, f32)],
    rng: &RngStrategy,
) -> Result<(u32, f32)> {
    if idx_probs.is_empty() {
        return Err(Error::Msg(
            "All probabilities are zero after filtering (top-k/top-p/min-p).".into(),
        ));
    }

    let weights: Vec<f32> = idx_probs.iter().map(|(_, p)| *p).collect();
    let distr = WeightedIndex::new(&weights).map_err(|e| {
        // Find the faulty element to report more specifically
        if let Some((i, p)) = weights
            .iter()
            .enumerate()
            .find(|(_, p)| !p.is_finite() || **p < 0.0)
        {
            Error::Msg(format!("Invalid prob at filtered index {i}: {p}"))
        } else {
            Error::Msg(format!("Failed to build sampler: {e}"))
        }
    })?;

    let selected = match rng {
        RngStrategy::ThreadLocal => THREAD_RNG.with(|cell| distr.sample(&mut *cell.borrow_mut())),
        RngStrategy::Shared(arc) => {
            let mut guard = arc
                .lock()
                .map_err(|_| Error::Msg("RNG lock poisoned".into()))?;
            distr.sample(&mut *guard)
        },
    };

    let (token, prob) = idx_probs[selected];
    Ok((token, prob))
}

/// Legacy: Sample from the entire `probs[]` — used for backward compatibility.
/// Less effective than `sample_from_filtered` when the vocabulary is large.
pub fn sample_from_full_probs(
    probs: &[f32],
    rng: &Arc<Mutex<ChaCha20Rng>>,
) -> Result<(u32, f32)> {
    let distr = WeightedIndex::new(probs).map_err(|e| Error::Msg(e.to_string()))?;
    let mut guard = rng
        .lock()
        .map_err(|_| Error::Msg("RNG lock poisoned".into()))?;
    let idx = distr.sample(&mut *guard);
    Ok((idx as u32, probs[idx]))
}

pub fn build_logprobs(
    token: u32,
    prob: f32,
    all_probs: &[f32],
    top_n: usize,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<Logprobs> {
    let logprob = if prob > 0.0 {
        prob.ln()
    } else {
        f32::NEG_INFINITY
    };

    let top_logprobs = if top_n > 0 {
        let mut pairs: Vec<(u32, f32)> = all_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        pairs.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(top_n);

        let mut result = Vec::with_capacity(top_n);
        for (t, p) in pairs {
            let bytes = if let Some(tok) = tokenizer {
                tok.decode(&[t], false).ok()
            } else {
                None
            };
            result.push(TopLogprob {
                token: t,
                logprob: if p > 0.0 {
                    p.ln()
                } else {
                    f32::NEG_INFINITY
                },
                bytes,
            });
        }
        Some(result)
    } else {
        None
    };

    let bytes = if let Some(tok) = tokenizer {
        tok.decode(&[token], false).ok()
    } else {
        None
    };

    Ok(Logprobs {
        token,
        logprob,
        bytes,
        top_logprobs,
    })
}

pub fn normalize(probs: &mut [f32]) -> Result<()> {
    let sum: f32 = probs
        .iter()
        .filter(|&&p| p.is_finite() && p > 0.0)
        .sum();
    if sum <= 0.0 {
        candle_core::bail!("All probabilities are zero — cannot normalize.");
    }
    for p in probs.iter_mut() {
        if p.is_finite() && *p > 0.0 {
            *p /= sum;
        } else {
            *p = 0.0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_from_filtered_single() {
        let idx_probs = vec![(5u32, 1.0f32)];
        let rng = RngStrategy::ThreadLocal;
        let (tok, prob) = sample_from_filtered(&idx_probs, &rng).unwrap();
        assert_eq!(tok, 5);
        assert!((prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_from_filtered_empty_errors() {
        let rng = RngStrategy::ThreadLocal;
        assert!(sample_from_filtered(&[], &rng).is_err());
    }

    #[test]
    fn test_normalize() {
        let mut p = vec![0.0f32, 0.0, 0.6, 0.0, 0.4];
        normalize(&mut p).unwrap();
        assert!((p[2] - 0.6).abs() < 1e-5);
        assert!((p[4] - 0.4).abs() < 1e-5);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_build_logprobs_natural_log() {
        let probs = vec![0.0f32, 1.0]; // token 1 = prob 1.0
        let lp = build_logprobs(1, 1.0, &probs, 0, None).unwrap();
        // ln(1.0) = 0.0
        assert!((lp.logprob - 0.0).abs() < 1e-5);
    }
}
