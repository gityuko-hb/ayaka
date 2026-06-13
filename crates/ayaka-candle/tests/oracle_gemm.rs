//! Oracle tests for gemm: cuBLASLt kernel vs candle-CPU f64 reference.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{gemm_new, gemm_ref};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};

/// (M, K, N) cases shared across dtypes. M=1 is the single-token decode
/// shape; 33 exercises a row count that is not a multiple of the tile size.
const SHAPES: [(usize, usize, usize); 4] =
    [(64, 64, 64), (1, 128, 256), (33, 256, 128), (128, 512, 64)];

/// Inputs for `a` are scaled by 1/sqrt(K) so outputs stay O(1) regardless of
/// K, keeping the shared absolute tolerances meaningful.
#[allow(clippy::too_many_arguments)]
fn check(
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    trans_a: bool,
    trans_b: bool,
    with_bias: bool,
    workspace_bytes: usize,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let a_dims = if trans_a {
        (k, m)
    } else {
        (m, k)
    };
    let b_dims = if trans_b {
        (n, k)
    } else {
        (k, n)
    };
    let scale = 1.0 / (k as f32).sqrt();
    let a = Tensor::randn(0f32, scale, a_dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let b = Tensor::randn(0f32, 1f32, b_dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = with_bias.then(|| {
        Tensor::randn(0f32, 1f32, n, &cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    });

    let want = gemm_ref(&a, &b, bias.as_ref(), trans_a, trans_b).unwrap();

    let a_gpu = a.to_device(&gpu).unwrap();
    let b_gpu = b.to_device(&gpu).unwrap();
    let bias_gpu = bias.as_ref().map(|t| t.to_device(&gpu).unwrap());
    let workspace =
        (workspace_bytes > 0).then(|| Tensor::zeros(workspace_bytes, DType::U8, &gpu).unwrap());
    let got = gemm_new(
        &a_gpu,
        &b_gpu,
        bias_gpu.as_ref(),
        trans_a,
        trans_b,
        workspace.as_ref(),
    )
    .unwrap();

    let what = format!(
        "gemm m={m} k={k} n={n} {dtype:?} trans_a={trans_a} trans_b={trans_b} bias={with_bias}"
    );
    assert_close(&got, &want, atol(dtype), &what);
}

#[test]
fn test_gemm_f32_nn_matches_oracle() {
    for (m, k, n) in SHAPES {
        check(m, k, n, DType::F32, false, false, false, 0);
    }
}

#[test]
fn test_gemm_f16_nn_matches_oracle() {
    for (m, k, n) in SHAPES {
        check(m, k, n, DType::F16, false, false, false, 0);
    }
}

#[test]
fn test_gemm_bf16_nn_matches_oracle() {
    for (m, k, n) in SHAPES {
        check(m, k, n, DType::BF16, false, false, false, 0);
    }
}

/// `x @ W^T` with W stored [N, K] — the linear-layer hot path.
#[test]
fn test_gemm_f32_nt_matches_oracle() {
    for (m, k, n) in SHAPES {
        check(m, k, n, DType::F32, false, true, false, 0);
    }
}

#[test]
fn test_gemm_bf16_nt_matches_oracle() {
    for (m, k, n) in SHAPES {
        check(m, k, n, DType::BF16, false, true, false, 0);
    }
}

#[test]
fn test_gemm_all_trans_combos_f32() {
    for (trans_a, trans_b) in [(false, false), (true, false), (false, true), (true, true)] {
        check(64, 64, 64, DType::F32, trans_a, trans_b, false, 0);
    }
}

#[test]
fn test_gemm_bias() {
    check(33, 256, 128, DType::F32, false, false, true, 0);
    check(33, 256, 128, DType::BF16, false, false, true, 0);
    check(33, 256, 128, DType::F32, false, true, true, 0);
}

#[test]
fn test_gemm_with_workspace() {
    check(128, 256, 128, DType::F32, false, false, false, 4 << 20);
}

#[test]
fn test_gemm_rejects_inner_dim_mismatch() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let a = Tensor::randn(0f32, 1f32, (64, 32), &gpu).unwrap();
    let b = Tensor::randn(0f32, 1f32, (64, 128), &gpu).unwrap();
    assert!(gemm_new(&a, &b, None, false, false, None).is_err());
}

#[test]
fn test_gemm_rejects_dtype_mismatch() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let a = Tensor::randn(0f32, 1f32, (64, 32), &gpu).unwrap();
    let b = Tensor::randn(0f32, 1f32, (32, 16), &gpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    assert!(gemm_new(&a, &b, None, false, false, None).is_err());
}

#[test]
fn test_gemm_rejects_cpu_tensors() {
    let cpu = Device::Cpu;
    let a = Tensor::randn(0f32, 1f32, (64, 32), &cpu).unwrap();
    let b = Tensor::randn(0f32, 1f32, (32, 16), &cpu).unwrap();
    assert!(gemm_new(&a, &b, None, false, false, None).is_err());
}

#[test]
fn test_gemm_rejects_bad_bias_length() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let a = Tensor::randn(0f32, 1f32, (64, 32), &gpu).unwrap();
    let b = Tensor::randn(0f32, 1f32, (32, 16), &gpu).unwrap();
    let bias = Tensor::randn(0f32, 1f32, 17, &gpu).unwrap();
    assert!(gemm_new(&a, &b, Some(&bias), false, false, None).is_err());
}
