use candle_core::{Result, Tensor};
use candle_nn::ops::softmax_last_dim;

use crate::logits::argmax_f32;

/// Results after applying temperature.

pub enum TemperatureOutput {
    /// temperature = None or < 1e-7 → sharp distribution, use argmax.
    Argmax(Vec<f32>),
    /// temperature > 0 → softmax probabilities.
    Probs(Vec<f32>),
}

/// Apply temperature scaling then softmax.
///
/// - `temperature = None` → returns argmax one-hot (no need for softmax).
/// - `temperature > 0` → `logits / T` then softmax.
pub fn apply_temperature(
    logits: Tensor,
    temperature: Option<f64>,
) -> Result<TemperatureOutput> {
    match temperature {
        None => {
            let raw: Vec<f32> = logits.to_vec1()?;
            // Create one-hot: argmax = 1.0, remaining = 0.0
            let best = argmax_f32(&raw) as usize;
            let mut one_hot = vec![0.0f32; raw.len()];
            one_hot[best] = 1.0;
            Ok(TemperatureOutput::Argmax(one_hot))
        },
        Some(t) => {
            let scaled = (&logits / t)?;
            let probs: Vec<f32> = softmax_last_dim(&scaled)?.to_vec1()?;
            Ok(TemperatureOutput::Probs(probs))
        },
    }
}

/// Lightweight: Just need scale + softmax -> Vec<f32>
/// Use internally for speculative decoding when probs are always needed.
pub fn scale_and_softmax(
    logits: &Tensor,
    temperature: f64,
) -> Result<Vec<f32>> {
    let scaled = (logits / temperature)?;
    softmax_last_dim(&scaled)?.to_vec1()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_argmax_temperature_none() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 0.3], 3, &Device::Cpu).unwrap();
        let out = apply_temperature(logits, None).unwrap();
        let TemperatureOutput::Argmax(v) = out else {
            panic!("expected Argmax")
        };
        assert_eq!(v[1], 1.0);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn test_softmax_temperature_one() {
        let logits = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], 3, &Device::Cpu).unwrap();
        let out = apply_temperature(logits, Some(1.0)).unwrap();
        let TemperatureOutput::Probs(v) = out else {
            panic!("expected Probs")
        };
        for p in &v {
            let diff = (p - 1.0 / 3.0).abs();
            assert!(diff < 1e-5, "prob={p}");
        }
    }
}
