/// Applies min-p filtering to an already-normalized probability distribution.
///
/// Min-p keeps tokens whose probability is at least `min_p * max_probability`.
/// Tokens below that threshold are set to `0.0` in `probs`.
///
/// Preconditions:
/// - `idx_probs` must contain `(token_index, probability)` pairs for `probs`.
/// - `idx_probs` should be sorted in descending probability order.
/// - `probs` should contain probabilities, not raw logits.
///
/// Notes:
/// - `min_p <= 0.0` disables filtering.
/// - `min_p >= 1.0` is clamped to `1.0`, keeping only tokens tied with the max probability.
/// - The caller must renormalize `probs` before sampling.
pub fn apply_min_p(
    probs: &mut [f32],
    idx_probs: &[(u32, f32)],
    min_p: f32,
) {
    if probs.is_empty() || idx_probs.is_empty() || !min_p.is_finite() || min_p <= 0.0 {
        return;
    }

    debug_assert!(
        idx_probs
            .windows(2)
            .all(|pair| pair[0].1 >= pair[1].1),
        "idx_probs must be sorted by descending probability"
    );
    debug_assert!(
        idx_probs
            .iter()
            .all(|&(idx, _)| (idx as usize) < probs.len()),
        "idx_probs contains an index outside probs"
    );

    let max_p = idx_probs[0].1;
    let threshold = max_p * min_p.min(1.0);

    for &(idx, prob) in idx_probs {
        if prob < threshold {
            probs[idx as usize] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(v: &[f32]) -> Vec<(u32, f32)> {
        let mut items: Vec<(u32, f32)> = v
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, prob)| (idx as u32, prob))
            .collect();
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        items
    }

    #[test]
    fn filters_probabilities_below_relative_threshold() {
        // max = 0.8, min_p = 0.1, threshold = 0.08.
        let mut probs = vec![0.8f32, 0.5, 0.05];
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 0.1);

        assert_eq!(probs, vec![0.8, 0.5, 0.0]);
    }

    #[test]
    fn keeps_probabilities_equal_to_threshold() {
        // max = 0.8, min_p = 0.125, threshold = 0.1.
        let mut probs = vec![0.8f32, 0.1, 0.099];
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 0.125);

        assert_eq!(probs, vec![0.8, 0.1, 0.0]);
    }

    #[test]
    fn keeps_only_max_probability_tokens_when_min_p_is_one() {
        let mut probs = vec![0.5f32, 0.5, 0.4, 0.1];
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 1.0);

        assert_eq!(probs, vec![0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn clamps_min_p_above_one_to_one() {
        let mut probs = vec![0.7f32, 0.6, 0.2];
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 2.0);

        assert_eq!(probs, vec![0.7, 0.0, 0.0]);
    }

    #[test]
    fn does_nothing_when_min_p_is_zero() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        let original = probs.clone();
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 0.0);

        assert_eq!(probs, original);
    }

    #[test]
    fn does_nothing_when_min_p_is_negative() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        let original = probs.clone();
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, -0.1);

        assert_eq!(probs, original);
    }

    #[test]
    fn does_nothing_when_min_p_is_nan() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        let original = probs.clone();
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, f32::NAN);

        assert_eq!(probs, original);
    }

    #[test]
    fn handles_empty_input() {
        let mut probs = Vec::<f32>::new();
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 0.1);

        assert!(probs.is_empty());
    }

    #[test]
    fn handles_single_token_input() {
        let mut probs = vec![1.0f32];
        let idx_probs = sorted(&probs);

        apply_min_p(&mut probs, &idx_probs, 0.5);

        assert_eq!(probs, vec![1.0]);
    }
}
