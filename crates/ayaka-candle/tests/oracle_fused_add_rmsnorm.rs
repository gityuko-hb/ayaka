//! Oracle tests for fused_add_rmsnorm: GPU kernel vs candle-CPU reference.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{fused_add_rmsnorm, fused_add_rmsnorm_new, fused_add_rmsnorm_ref};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};

const EPS: f32 = 1e-5;

fn check(
    dims: &[usize],
    dtype: DType,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let hidden = *dims.last().unwrap();
    let input = Tensor::randn(0f32, 1f32, dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let residual = Tensor::randn(0f32, 1f32, dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let weight = Tensor::randn(0f32, 1f32, hidden, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let (want_normed, want_added) = fused_add_rmsnorm_ref(&input, &residual, &weight, EPS).unwrap();
    let (got_normed, got_added) = fused_add_rmsnorm_new(
        &input.to_device(&gpu).unwrap(),
        &residual.to_device(&gpu).unwrap(),
        &weight.to_device(&gpu).unwrap(),
        EPS,
    )
    .unwrap();

    let what = format!("fused_add_rmsnorm {dims:?} {dtype:?}");
    assert_close(
        &got_normed,
        &want_normed,
        atol(dtype),
        &format!("{what} normed"),
    );
    assert_close(
        &got_added,
        &want_added,
        atol(dtype),
        &format!("{what} added"),
    );
}

#[test]
fn test_fused_add_rmsnorm_f32_matches_oracle() {
    for dims in [&[1usize, 16][..], &[8, 128], &[33, 1024], &[4, 4096]] {
        check(dims, DType::F32);
    }
}

#[test]
fn test_fused_add_rmsnorm_f16_matches_oracle() {
    for dims in [&[8usize, 128][..], &[4, 4096]] {
        check(dims, DType::F16);
    }
}

#[test]
fn test_fused_add_rmsnorm_bf16_matches_oracle() {
    for dims in [&[8usize, 128][..], &[4, 4096]] {
        check(dims, DType::BF16);
    }
}

#[test]
fn test_fused_add_rmsnorm_in_place() {
    // The production pattern: out aliases input, residual_out aliases residual.
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let input_cpu = Tensor::randn(0f32, 1f32, (8, 256), &cpu).unwrap();
    let residual_cpu = Tensor::randn(0f32, 1f32, (8, 256), &cpu).unwrap();
    let weight_cpu = Tensor::randn(0f32, 1f32, 256, &cpu).unwrap();

    let input = input_cpu.to_device(&gpu).unwrap();
    let residual = residual_cpu.to_device(&gpu).unwrap();
    let weight = weight_cpu.to_device(&gpu).unwrap();

    fused_add_rmsnorm(&input, &residual, &input, &residual, &weight, EPS).unwrap();

    let (want_normed, want_added) =
        fused_add_rmsnorm_ref(&input_cpu, &residual_cpu, &weight_cpu, EPS).unwrap();
    assert_close(&input, &want_normed, atol(DType::F32), "in-place normed");
    assert_close(&residual, &want_added, atol(DType::F32), "in-place added");
}

#[test]
fn test_fused_add_rmsnorm_rejects_shape_mismatch() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let input = Tensor::randn(0f32, 1f32, (4, 64), &gpu).unwrap();
    let residual = Tensor::randn(0f32, 1f32, (4, 32), &gpu).unwrap();
    let weight = Tensor::randn(0f32, 1f32, 64, &gpu).unwrap();
    assert!(fused_add_rmsnorm_new(&input, &residual, &weight, EPS).is_err());
}
