//! Oracle tests for silu_and_mul: GPU kernel vs candle-CPU reference.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{silu_and_mul_new, silu_and_mul_ref};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};

fn check(
    dims: &[usize],
    dtype: DType,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let input = Tensor::randn(0f32, 1f32, dims, &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let want = silu_and_mul_ref(&input).unwrap();
    let got = silu_and_mul_new(&input.to_device(&gpu).unwrap()).unwrap();

    let what = format!("silu_and_mul {dims:?} {dtype:?}");
    assert_close(&got, &want, atol(dtype), &what);
}

#[test]
fn test_silu_and_mul_f32_matches_oracle() {
    for dims in [&[1usize, 32][..], &[8, 256], &[33, 2048], &[4, 8192]] {
        check(dims, DType::F32);
    }
}

#[test]
fn test_silu_and_mul_f16_matches_oracle() {
    for dims in [&[8usize, 256][..], &[4, 8192]] {
        check(dims, DType::F16);
    }
}

#[test]
fn test_silu_and_mul_bf16_matches_oracle() {
    for dims in [&[8usize, 256][..], &[4, 8192]] {
        check(dims, DType::BF16);
    }
}

#[test]
fn test_silu_and_mul_3d_input() {
    check(&[2, 5, 512], DType::F32);
}

#[test]
fn test_silu_and_mul_rejects_odd_last_dim() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let input = Tensor::randn(0f32, 1f32, (4, 63), &gpu).unwrap();
    assert!(silu_and_mul_new(&input).is_err());
}

#[test]
fn test_silu_and_mul_rejects_cpu_tensors() {
    let cpu = Device::Cpu;
    let input = Tensor::randn(0f32, 1f32, (4, 64), &cpu).unwrap();
    assert!(silu_and_mul_new(&input).is_err());
}
