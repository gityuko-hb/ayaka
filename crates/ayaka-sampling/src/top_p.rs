
pub fn apply_top_p(probs: &mut [f32], idx_probs: &[(u32, f32)], p: f32) {
    if p <= 0.0 || p >= 1.0 {
        return;
    }
    let mut cumsum = 0.0f32;
    for &(idx, prob) in idx_probs {
        if cumsum >= p {
            probs[idx as usize] = 0.0;
        } else {
            cumsum += prob;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sorted(v: &[f32]) -> Vec<(u32, f32)> {
        let mut idx: Vec<(u32, f32)> = v.iter().copied().enumerate().map(|(i,p)|(i as u32,p)).collect();
        idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        idx
    }

    #[test]
    fn test_top_p_cuts_tail() {
        // probs = [0.6, 0.3, 0.1], p=0.9 → giữ 0.6+0.3, cắt 0.1
        let mut probs = vec![0.6f32, 0.3, 0.1];
        let sorted = make_sorted(&probs);
        apply_top_p(&mut probs, &sorted, 0.9);
        assert!(probs[2] == 0.0);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
    }

    #[test]
    fn test_top_p_noop_when_one() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        let orig = probs.clone();
        let sorted = make_sorted(&probs);
        apply_top_p(&mut probs, &sorted, 1.0);
        assert_eq!(probs, orig);
    }

    #[test]
    fn test_top_p_noop_when_zero() {
        let mut probs = vec![0.5f32, 0.5];
        let orig = probs.clone();
        let sorted = make_sorted(&probs);
        apply_top_p(&mut probs, &sorted, 0.0);
        assert_eq!(probs, orig);
    }
}