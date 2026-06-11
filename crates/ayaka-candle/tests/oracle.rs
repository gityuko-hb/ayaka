//! Numeric oracle harness: native GPU kernels vs the pure-candle CPU
//! reference, compared element-wise under a per-dtype tolerance.
//!
//! Reusable pattern for every op after RMSNorm: build inputs on CPU, cast to
//! the target dtype, run the reference on CPU and the kernel on CUDA, then
//! `assert_close` with `atol(dtype)`.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{rmsnorm_new, rmsnorm_ref};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};

const EPS: f32 = 1e-5;

/// Compare kernel output against the CPU oracle for one shape/dtype/eps.
fn check_rmsnorm(
    dims: &[usize],
    dtype: DType,
    eps: f32,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let hidden = *dims.last().unwrap();
    let input = Tensor::randn(0f32, 1f32, dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let weight = Tensor::randn(0f32, 1f32, hidden, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let want = rmsnorm_ref(&input, &weight, eps).unwrap();
    let got = rmsnorm_new(
        &input.to_device(&gpu).unwrap(),
        &weight.to_device(&gpu).unwrap(),
        eps,
    )
    .unwrap()
    .to_device(&cpu)
    .unwrap();

    let what = format!("rmsnorm {dims:?} {dtype:?} eps={eps}");
    assert_close(&got, &want, atol(dtype), &what);
}

#[test]
fn test_rmsnorm_f32_matches_oracle() {
    for dims in [&[1usize, 16][..], &[8, 128], &[33, 1024], &[4, 4096]] {
        check_rmsnorm(dims, DType::F32, EPS);
    }
}

#[test]
fn test_rmsnorm_f16_matches_oracle() {
    for dims in [&[1usize, 16][..], &[8, 128], &[33, 1024], &[4, 4096]] {
        check_rmsnorm(dims, DType::F16, EPS);
    }
}

#[test]
fn test_rmsnorm_bf16_matches_oracle() {
    for dims in [&[1usize, 16][..], &[8, 128], &[33, 1024], &[4, 4096]] {
        check_rmsnorm(dims, DType::BF16, EPS);
    }
}

#[test]
fn test_rmsnorm_3d_input() {
    check_rmsnorm(&[2, 3, 512], DType::F32, EPS);
}

#[test]
fn test_rmsnorm_eps_variants() {
    check_rmsnorm(&[8, 256], DType::F32, 1e-6);
    check_rmsnorm(&[8, 256], DType::F32, 1e-2);
}

#[test]
fn test_rmsnorm_rejects_cpu_tensors() {
    let cpu = Device::Cpu;
    let input = Tensor::randn(0f32, 1f32, (4, 64), &cpu).unwrap();
    let weight = Tensor::randn(0f32, 1f32, 64, &cpu).unwrap();
    assert!(rmsnorm_new(&input, &weight, EPS).is_err());
}

#[test]
fn test_rmsnorm_rejects_dtype_mismatch() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let input = Tensor::randn(0f32, 1f32, (4, 64), &gpu).unwrap();
    let weight = Tensor::randn(0f32, 1f32, 64, &gpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    assert!(rmsnorm_new(&input, &weight, EPS).is_err());
}

#[test]
fn test_rmsnorm_rejects_weight_length_mismatch() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let input = Tensor::randn(0f32, 1f32, (4, 64), &gpu).unwrap();
    let weight = Tensor::randn(0f32, 1f32, 32, &gpu).unwrap();
    assert!(rmsnorm_new(&input, &weight, EPS).is_err());
}

#[test]
fn test_rmsnorm_rejects_non_contiguous_input() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let input = Tensor::randn(0f32, 1f32, (64, 4), &gpu)
        .unwrap()
        .t()
        .unwrap();
    let weight = Tensor::randn(0f32, 1f32, 64, &gpu).unwrap();
    assert!(rmsnorm_new(&input, &weight, EPS).is_err());
}
