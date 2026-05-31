//! Custom logits processor chain.
//!
//! Allows injecting arbitrary transforms into the pipeline before sampling.
//! Processors run in order — the output of processor n is the input of processor n+1.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::logits::LogitsProcessor;

// ── ProcessorChain
/// List of processors running sequentially.
/// Easier to extend than `Vec<Arc<dyn LogitsProcessor>>` in the original code because
/// metadata, debug info, or conditional bypass can be added later.
#[derive(Clone, Default)]
pub struct ProcessorChain {
    processors: Vec<Arc<dyn LogitsProcessor>>,
}

impl ProcessorChain {
    pub fn new(processors: Vec<Arc<dyn LogitsProcessor>>) -> Self {
        Self { processors }
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Run the entire chain — each processor receives logits from the previous processor.
    pub fn apply(
        &self,
        mut logits: Tensor,
        context: &[u32],
    ) -> Result<Tensor> {
        for p in &self.processors {
            logits = p.apply(&logits, context)?;
        }
        Ok(logits)
    }
}

// ── Logits Bias ──────────────────────────────────────────────────────────────────────────
/// Apply `logits_bias` map to logits — add bias to specific token IDs.
///
/// Used to force/suppress specific tokens (e.g., ban EOS, boost specific tokens).
/// Separated from `ProcessorChain` as this is a simple user-facing feature,
/// no full processor overhead required.
pub fn apply_logits_bias(
    logits: &Tensor,
    bias: &HashMap<u32, f32>,
    device: &Device,
) -> Result<Tensor> {
    if bias.is_empty() {
        return Ok(logits.clone());
    }

    let vocab = logits.shape().dims1()?;
    let mut delta = vec![0.0f32; vocab];

    for (&token_id, &bias_val) in bias {
        if (token_id as usize) < vocab {
            delta[token_id as usize] = bias_val;
        }
    }

    let delta_tensor = Tensor::from_slice(&delta, vocab, device)?;
    logits + delta_tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_empty_chain_is_noop() {
        let chain = ProcessorChain::default();
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
        let out = chain.apply(logits.clone(), &[]).unwrap();
        let a: Vec<f32> = logits.to_vec1().unwrap();
        let b: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_chain_applies_processor_in_order() {
        // Processor 1: +1 to all, Processor 2: *2 to all
        // → (logits + 1) * 2
        let p1: Arc<dyn LogitsProcessor> = Arc::new(|logits: &Tensor, _: &[u32]| logits + 1.0f64);
        let p2: Arc<dyn LogitsProcessor> = Arc::new(|logits: &Tensor, _: &[u32]| logits * 2.0f64);
        let chain = ProcessorChain::new(vec![p1, p2]);
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
        let out: Vec<f32> = chain
            .apply(logits, &[])
            .unwrap()
            .to_vec1()
            .unwrap();
        // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        assert!((out[0] - 4.0).abs() < 1e-4);
        assert!((out[1] - 6.0).abs() < 1e-4);
        assert!((out[2] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_logits_bias_adds_delta() {
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], 3, &Device::Cpu).unwrap();
        let mut bias = HashMap::new();
        bias.insert(1u32, 5.0f32);
        let out: Vec<f32> = apply_logits_bias(&logits, &bias, &Device::Cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_logits_bias_empty_is_noop() {
        let logits = Tensor::from_vec(vec![1.0f32, 2.0], 2, &Device::Cpu).unwrap();
        let out = apply_logits_bias(&logits, &HashMap::new(), &Device::Cpu).unwrap();
        let a: Vec<f32> = logits.to_vec1().unwrap();
        let b: Vec<f32> = out.to_vec1().unwrap();
        assert_eq!(a, b);
    }
}
