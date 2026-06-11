//! Shared oracle-test helpers: per-dtype tolerances and element-wise
//! comparison. Every op's oracle test reuses this harness.

use candle_core::{DType, Tensor};

pub fn atol(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-4,
        DType::F16 => 1e-2,
        DType::BF16 => 8e-2,
        dt => panic!("no tolerance defined for {dt:?}"),
    }
}

pub fn assert_close(
    got: &Tensor,
    want: &Tensor,
    atol: f32,
    what: &str,
) {
    let got: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let want: Vec<f32> = want
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut max_diff = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_diff = max_diff.max((g - w).abs());
    }
    assert!(
        max_diff <= atol,
        "{what}: max abs diff {max_diff} exceeds atol {atol}"
    );
}
