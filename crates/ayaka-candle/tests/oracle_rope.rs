//! Oracle tests for rope (neox + gptj): in-place GPU kernel vs candle-CPU
//! reference, including partial rotary dims and grouped-query head counts.

#![cfg(feature = "cuda")]

mod common;

use ayaka_candle::ops::{RopeLayout, cos_sin_cache, rope, rope_ref};
use candle_core::{DType, Device, Tensor};
use common::{assert_close, atol};

const MAX_POS: usize = 128;
const THETA: f32 = 10_000.0;

#[allow(clippy::too_many_arguments)]
fn check(
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    dtype: DType,
    layout: RopeLayout,
) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("CUDA device 0");

    let q_cpu = Tensor::randn(0f32, 1f32, (tokens, heads, head_dim), &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k_cpu = Tensor::randn(0f32, 1f32, (tokens, kv_heads, head_dim), &cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let positions_cpu =
        Tensor::from_iter((0..tokens).map(|t| ((t * 37 + 11) % MAX_POS) as i64), &cpu).unwrap();
    let cache_cpu = cos_sin_cache(MAX_POS, rot_dim, THETA, &cpu).unwrap();

    let want_q = rope_ref(&q_cpu, &positions_cpu, &cache_cpu, layout).unwrap();
    let want_k = rope_ref(&k_cpu, &positions_cpu, &cache_cpu, layout).unwrap();

    let q = q_cpu.to_device(&gpu).unwrap();
    let k = k_cpu.to_device(&gpu).unwrap();
    let positions = positions_cpu.to_device(&gpu).unwrap();
    let cache = cache_cpu.to_device(&gpu).unwrap();
    rope(&q, &k, &positions, &cache, layout).unwrap();

    let what = format!(
        "rope t={tokens} h={heads}/{kv_heads} hd={head_dim} rot={rot_dim} {dtype:?} {layout:?}"
    );
    assert_close(&q, &want_q, atol(dtype), &format!("{what} query"));
    assert_close(&k, &want_k, atol(dtype), &format!("{what} key"));
}

#[test]
fn test_rope_neox_f32_matches_oracle() {
    check(7, 4, 4, 64, 64, DType::F32, RopeLayout::Neox);
    check(33, 8, 8, 128, 128, DType::F32, RopeLayout::Neox);
}

#[test]
fn test_rope_gptj_f32_matches_oracle() {
    check(7, 4, 4, 64, 64, DType::F32, RopeLayout::GptJ);
    check(33, 8, 8, 128, 128, DType::F32, RopeLayout::GptJ);
}

#[test]
fn test_rope_partial_rot_dim() {
    check(9, 4, 4, 64, 32, DType::F32, RopeLayout::Neox);
    check(9, 4, 4, 64, 32, DType::F32, RopeLayout::GptJ);
}

#[test]
fn test_rope_grouped_kv_heads() {
    check(11, 8, 2, 64, 64, DType::F32, RopeLayout::Neox);
}

#[test]
fn test_rope_f16_matches_oracle() {
    check(7, 4, 4, 64, 64, DType::F16, RopeLayout::Neox);
    check(7, 4, 4, 64, 64, DType::F16, RopeLayout::GptJ);
}

#[test]
fn test_rope_bf16_matches_oracle() {
    check(7, 4, 4, 64, 64, DType::BF16, RopeLayout::Neox);
}

#[test]
fn test_rope_rejects_bad_positions_dtype() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let q = Tensor::randn(0f32, 1f32, (4, 2, 64), &gpu).unwrap();
    let k = Tensor::randn(0f32, 1f32, (4, 2, 64), &gpu).unwrap();
    let positions = Tensor::zeros(4, DType::U32, &gpu).unwrap();
    let cache = cos_sin_cache(MAX_POS, 64, THETA, &gpu).unwrap();
    assert!(rope(&q, &k, &positions, &cache, RopeLayout::Neox).is_err());
}

#[test]
fn test_rope_rejects_rot_dim_exceeding_head_dim() {
    let gpu = Device::new_cuda(0).expect("CUDA device 0");
    let q = Tensor::randn(0f32, 1f32, (4, 2, 32), &gpu).unwrap();
    let k = Tensor::randn(0f32, 1f32, (4, 2, 32), &gpu).unwrap();
    let positions = Tensor::zeros(4, DType::I64, &gpu).unwrap();
    let cache = cos_sin_cache(MAX_POS, 64, THETA, &gpu).unwrap();
    assert!(rope(&q, &k, &positions, &cache, RopeLayout::Neox).is_err());
}
