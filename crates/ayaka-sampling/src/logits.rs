use candle_core::{Result, Tensor};

/// O(n) argmax trên Vec<f32> — tránh tạo Tensor khi không cần GPU.
#[inline]
pub fn argmax_f32(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Partial sort top-k: O(n) select + O(k log k) sort, instead of O(n log n) full sort.
///
/// Returns Vec<(token_id, prob)> sorted descending.
/// If `zero_rest = true`, zero out elements outside top-k in `probs`.
pub fn partial_sort_top_k(
    probs: &mut [f32],
    k: usize,
    zero_rest: bool,
) -> Vec<(u32, f32)> {
    let n = probs.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut idx_probs: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| (i, probs[i as usize]))
        .collect();
    let k = k.min(n);

    if k < n {
        idx_probs.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if zero_rest {
            for (idx, _) in &idx_probs[k..] {
                probs[*idx as usize] = 0.0;
            }
        }
        idx_probs.truncate(k);
    }
    idx_probs.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx_probs
}

/// Allows injecting custom transform into logit pipeline.
///
/// # For example
/// ```rust
/// use ayaka_sampling::logits::LogitsProcessor;
/// use candle_core::{Result, Tensor};
///
/// // Closure form:
/// let clamp: Box<dyn LogitsProcessor> =
/// Box::new(|logits: &Tensor, _ctx: &[u32]| logits.clamp(-10f32, 10f32));
///
/// // Structure format:
/// struct MyProcessor;
/// impl LogitsProcessor for MyProcessor {
/// fn apply(&self, logits: &Tensor, _ctx: &[u32]) -> Result<Tensor> {
/// logits.clamp(-5f32, 5f32)
/// }
/// }
/// ```
pub trait LogitsProcessor: Send + Sync {
    fn apply(
        &self,
        logits: &Tensor,
        context: &[u32],
    ) -> Result<Tensor>;
}

impl<F> LogitsProcessor for F
where
    F: Fn(&Tensor, &[u32]) -> Result<Tensor> + Send + Sync,
{
    fn apply(
        &self,
        logits: &Tensor,
        context: &[u32],
    ) -> Result<Tensor> {
        self(logits, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let v = vec![0.1f32, 0.9, 0.3, 0.7];
        assert_eq!(argmax_f32(&v), 1);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax_f32(&[42.0f32]), 0);
    }

    #[test]
    fn test_partial_sort_top_k_basic() {
        let mut probs = vec![0.1f32, 0.5, 0.3, 0.8, 0.05];
        let top = partial_sort_top_k(&mut probs, 2, true);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 3); // 0.8
        assert_eq!(top[1].0, 1); // 0.5
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[4], 0.0);
    }

    #[test]
    fn test_partial_sort_top_k_k_ge_n() {
        let mut probs = vec![0.2f32, 0.8];
        let top = partial_sort_top_k(&mut probs, 10, false);
        assert_eq!(top.len(), 2);
    }
}
