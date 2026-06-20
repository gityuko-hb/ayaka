//! Oracle tests for the W4A16 Marlin kernel vs the candle reference.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{quant_gemm_marlin_new, quant_gemm_ref};
use ayaka_quant::{QuantScheme, RepackedWeights, repack_to_marlin};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};
use half::f16;

/// Deterministic W4A16 weights in the de-interleaved `[N, K/2]` layout.
fn build_weights(
    n: usize,
    k: usize,
    group_size: usize,
    with_mins: bool,
) -> RepackedWeights {
    let n_groups = k / group_size;
    let mut qs = vec![0u8; n * k / 2];
    for (i, b) in qs.iter_mut().enumerate() {
        *b = ((i * 37 + 11) & 0xFF) as u8; // arbitrary but fixed nibbles
    }
    let scales: Vec<f16> = (0..n * n_groups)
        .map(|i| f16::from_f32(0.05 + 0.01 * ((i % 7) as f32)))
        .collect();
    let mins: Vec<f16> = if with_mins {
        (0..n * n_groups)
            .map(|i| f16::from_f32(-0.1 + 0.02 * ((i % 5) as f32)))
            .collect()
    } else {
        Vec::new()
    };
    RepackedWeights {
        qs,
        scales,
        mins,
        out_features: n,
        in_features: k,
        group_size,
        scheme: QuantScheme::Q4K,
    }
}

#[allow(clippy::too_many_arguments)]
fn check(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    dtype: DType,
    with_mins: bool,
    with_bias: bool,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let rw = build_weights(n, k, group_size, with_mins);

    // Activations scaled by 1/sqrt(K) so outputs stay O(1) for the shared atols.
    let scale = 1.0 / (k as f32).sqrt();
    let a = Tensor::randn(0f32, scale, (m, k), &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // Reference inputs: b_quant in `[N, K/2]`; scales/mins as F16 (the ref reads f16).
    let b_quant_cpu = Tensor::from_vec(rw.qs.clone(), (n, k / 2), &cpu).unwrap();
    let b_scales_f16 = Tensor::from_vec(rw.scales.clone(), (n, k / group_size), &cpu).unwrap();
    let b_mins_f16 =
        with_mins.then(|| Tensor::from_vec(rw.mins.clone(), (n, k / group_size), &cpu).unwrap());
    let bias = with_bias.then(|| {
        Tensor::randn(0f32, 1f32, n, &cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    });

    let want = quant_gemm_ref(
        &a,
        &b_quant_cpu,
        &b_scales_f16,
        b_mins_f16.as_ref(),
        bias.as_ref(),
        group_size,
    )
    .unwrap();

    // GPU Marlin: weights in Marlin tile layout; scales/mins cast to `dtype`.
    let marlin = repack_to_marlin(&rw).unwrap();
    let a_g = a.to_device(&gpu).unwrap();
    let bq_g = Tensor::from_vec(marlin.qs.clone(), (n, k / 2), &gpu).unwrap();
    let bs_g = b_scales_f16
        .to_dtype(dtype)
        .unwrap()
        .to_device(&gpu)
        .unwrap();
    let bm_g = b_mins_f16.as_ref().map(|t| {
        t.to_dtype(dtype)
            .unwrap()
            .to_device(&gpu)
            .unwrap()
    });
    let bias_g = bias.as_ref().map(|t| t.to_device(&gpu).unwrap());

    let got = quant_gemm_marlin_new(
        &a_g,
        &bq_g,
        &bs_g,
        bm_g.as_ref(),
        bias_g.as_ref(),
        group_size,
        None,
    )
    .unwrap();

    let what = format!(
        "marlin m={m} k={k} n={n} g={group_size} {dtype:?} mins={with_mins} bias={with_bias}"
    );
    assert_close(&got, &want, atol(dtype), &what);
}

/// (M, K, N, group): M=1 is decode; M=17 exercises a partial M-tile.
const CASES: [(usize, usize, usize, usize); 4] = [
    (16, 32, 16, 16),
    (1, 256, 32, 32),
    (17, 256, 64, 128),
    (128, 512, 32, 128),
];

#[test]
fn marlin_f16_matches_oracle() {
    for (m, k, n, g) in CASES {
        check(m, k, n, g, DType::F16, true, false);
    }
}

#[test]
fn marlin_bf16_matches_oracle() {
    for (m, k, n, g) in CASES {
        check(m, k, n, g, DType::BF16, true, false);
    }
}

#[test]
fn marlin_symmetric_no_mins() {
    check(64, 256, 64, 128, DType::F16, false, false);
    check(64, 256, 64, 128, DType::BF16, false, false);
}

#[test]
fn marlin_with_bias() {
    check(33, 256, 64, 128, DType::F16, true, true);
    check(33, 256, 64, 128, DType::BF16, true, true);
}

#[test]
fn marlin_rejects_unaligned_n() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let a = Tensor::randn(0f32, 1f32, (16, 32), &gpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let bq = Tensor::zeros((24, 16), DType::U8, &gpu).unwrap(); // N=24 not %16
    let bs = Tensor::zeros((24, 2), DType::F16, &gpu).unwrap();
    assert!(quant_gemm_marlin_new(&a, &bq, &bs, None, None, 16, None).is_err());
}

#[test]
fn quant_linear_forward_uses_marlin_oracle() {
    use ayaka_candle::ops::QuantLinear;

    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let (n, k, group_size) = (64usize, 256usize, 128usize);
    let rw = build_weights(n, k, group_size, true);

    // Oracle from the [N, K/2] form.
    let cpu = Device::Cpu;
    let a_cpu = Tensor::randn(0f32, 1.0 / (k as f32).sqrt(), (2, k), &cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let b_quant_cpu = Tensor::from_vec(rw.qs.clone(), (n, k / 2), &cpu).unwrap();
    let b_scales_f16 = Tensor::from_vec(rw.scales.clone(), (n, k / group_size), &cpu).unwrap();
    let b_mins_f16 = Tensor::from_vec(rw.mins.clone(), (n, k / group_size), &cpu).unwrap();
    let want = quant_gemm_ref(
        &a_cpu,
        &b_quant_cpu,
        &b_scales_f16,
        Some(&b_mins_f16),
        None,
        group_size,
    )
    .unwrap();

    let layer = QuantLinear::from_repacked(&rw, None, &gpu).unwrap();
    let x = a_cpu
        .to_device(&gpu)
        .unwrap()
        .reshape((1, 2, k))
        .unwrap();
    let got = layer
        .forward(&x)
        .unwrap()
        .reshape((2, n))
        .unwrap();
    assert_close(&got, &want, atol(DType::F16), "quant_linear marlin forward");
}
