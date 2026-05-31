use crate::logits::partial_sort_top_k;

/// Apply top-k filter to in-place `probs`.
/// - `k = 0` → no filter (pass-through).
/// - Use partial sort O(n) + O(k log k), zero out the rest.E
/// - Return sorted top-k pairs for top_p / min_p to use again — avoid resorting.
pub fn apply_top_k(
    probs: &mut [f32],
    k: usize,
) -> Vec<(u32, f32)> {
    if k == 0 {
        // k=0 -> no limit — returns the sorted total.
        let effective_k = probs.len();
        return partial_sort_top_k(probs, effective_k, false);
    }
    partial_sort_top_k(probs, k, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_filters_correctly() {
        let mut probs = vec![0.1f32, 0.5, 0.3, 0.8, 0.05];
        let top = apply_top_k(&mut probs, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 3); // 0.8
        assert_eq!(top[1].0, 1); // 0.5
        // Những phần tử ngoài top-2 phải bị zero
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[4], 0.0);
    }

    #[test]
    fn test_top_k_zero_means_no_filter() {
        let mut probs = vec![0.2f32, 0.5, 0.3];
        let top = apply_top_k(&mut probs, 0);
        assert_eq!(top.len(), 3);
        // Không zero out gì
        assert!(probs[0] > 0.0);
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut probs = vec![0.4f32, 0.6];
        let top = apply_top_k(&mut probs, 100);
        assert_eq!(top.len(), 2);
    }
}
