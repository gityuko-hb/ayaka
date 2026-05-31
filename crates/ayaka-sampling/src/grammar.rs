//! Grammar constraint for structured generation.
//!
//! Completely separate from sampling logic — this module only knows about
//! "Is the token allowed?" and "Which mask should be used for resample?".

use candle_core::{Device, Result, Tensor};

// GrammarMask trait
/// Abstraction for any grammar constraint engine
///
/// (Illguidance, GBNF, FSM regex, ...).
///
/// Decoupled from specific Illguidance — easy to mock in tests.
pub trait GrammarMask: Send + Sync {
    /// Check is this token allowed?
    fn is_allowed(
        &self,
        token: u32,
    ) -> bool;

    /// Calculate bias vector: 0.0 for allowed tokens, -inf for forbidden tokens.
    /// The result is added to logits before resample.
    fn bias_vector(
        &self,
        vocab_size: usize,
    ) -> Vec<f32>;

    /// Is the grammar complete (EOS or structure complete)?
    fn is_stopped(&self) -> bool;

    /// Consume token after example — advanced grammar state.
    fn consume(
        &mut self,
        token: u32,
    ) -> Result<()>;
}

// Grammar pipeline
/// Grammar check results for a sampled token.
pub enum GrammarCheck {
    /// Valid token, no resample needed.
    Allowed,

    // Forbidden token — resample needed with this bias tensor.
    Disallowed(Tensor),
}

/// Check if the sampled token passes the grammar constraint.
/// If not, build a bias tensor to resample.
pub fn check_token<G: GrammarMask>(
    mask: &G,
    token: u32,
    vocab_size: usize,
    device: &Device,
) -> Result<GrammarCheck> {
    if mask.is_stopped() || mask.is_allowed(token) {
        return Ok(GrammarCheck::Allowed);
    }

    let bias = mask.bias_vector(vocab_size);
    let tensor = Tensor::from_slice(&bias, vocab_size, device)?;
    Ok(GrammarCheck::Disallowed(tensor))
}

/// Apply a bias tensor to logits: `new_logits = logits + bias`.
/// Use when `check_token` returns `Disallowed`.
pub fn apply_grammar_bias(
    logits: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    logits + bias
}

// No-op implementation for testing
/// Grammar mask always allows all tokens — use when there is no grammar constraint.
pub struct NoGrammar;

impl GrammarMask for NoGrammar {
    fn is_allowed(
        &self,
        _token: u32,
    ) -> bool {
        true
    }
    fn bias_vector(
        &self,
        vocab_size: usize,
    ) -> Vec<f32> {
        vec![0.0; vocab_size]
    }
    fn is_stopped(&self) -> bool {
        false
    }
    fn consume(
        &mut self,
        _token: u32,
    ) -> Result<()> {
        Ok(())
    }
}

/// Grammar masks only allow a fixed set of tokens — for use in tests.
pub struct AllowListGrammar {
    pub allowed: std::collections::HashSet<u32>,
    pub stopped: bool,
}

impl GrammarMask for AllowListGrammar {
    fn is_allowed(
        &self,
        token: u32,
    ) -> bool {
        self.allowed.contains(&token)
    }
    fn bias_vector(
        &self,
        vocab_size: usize,
    ) -> Vec<f32> {
        (0..vocab_size as u32)
            .map(|t| {
                if self.allowed.contains(&t) {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect()
    }
    fn is_stopped(&self) -> bool {
        self.stopped
    }
    fn consume(
        &mut self,
        _token: u32,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_no_grammar_always_allows() {
        let g = NoGrammar;
        assert!(g.is_allowed(0));
        assert!(g.is_allowed(99999));
        assert!(!g.is_stopped());
    }

    #[test]
    fn test_allowlist_grammar_check() {
        let g = AllowListGrammar {
            allowed: [1u32, 3, 5].into_iter().collect(),
            stopped: false,
        };
        assert!(g.is_allowed(1));
        assert!(!g.is_allowed(2));
    }

    #[test]
    fn test_check_token_allowed() {
        let g = AllowListGrammar {
            allowed: [42u32].into_iter().collect(),
            stopped: false,
        };
        let result = check_token(&g, 42, 100, &Device::Cpu).unwrap();
        assert!(matches!(result, GrammarCheck::Allowed));
    }

    #[test]
    fn test_check_token_disallowed_builds_tensor() {
        let g = AllowListGrammar {
            allowed: [1u32].into_iter().collect(),
            stopped: false,
        };
        let result = check_token(&g, 0, 4, &Device::Cpu).unwrap();
        assert!(matches!(result, GrammarCheck::Disallowed(_)));
    }

    #[test]
    fn test_stopped_grammar_always_allows() {
        let g = AllowListGrammar {
            allowed: std::collections::HashSet::new(), // nothing allowed
            stopped: true,                             // Key stopped → always allowed
        };
        let result = check_token(&g, 99, 100, &Device::Cpu).unwrap();
        assert!(matches!(result, GrammarCheck::Allowed));
    }
}
