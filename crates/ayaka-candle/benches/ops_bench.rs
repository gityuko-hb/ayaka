//! Kernel benchmarks for all native ops (requires `--features cuda` and a
//! CUDA device). Shapes approximate a 7B-class decoder layer.
//!
//! Run with: `cargo bench -p ayaka-candle --features cuda`

use ayaka_candle::ops::{
    RopeLayout, cos_sin_cache, fused_add_rmsnorm, rmsnorm, rope, silu_and_mul,
};
use candle_core::{DType, Device, Tensor};
use criterion::{Criterion, criterion_group, criterion_main};

const DTYPES: [(DType, &str); 2] = [(DType::F32, "f32"), (DType::F16, "f16")];

fn randn(
    dims: (usize, usize),
    dtype: DType,
    device: &Device,
) -> Tensor {
    Tensor::randn(0f32, 1f32, dims, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn bench_rmsnorm(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    for (dtype, name) in DTYPES {
        let input = randn((256, 4096), dtype, &device);
        let weight = randn((1, 4096), dtype, &device)
            .squeeze(0)
            .unwrap();
        let out = input.zeros_like().unwrap();

        c.bench_function(&format!("ayaka_rmsnorm_{name}_256x4096"), |b| {
            b.iter(|| {
                rmsnorm(&out, &input, &weight, 1e-5).unwrap();
                device.synchronize().unwrap();
            })
        });
    }
}

fn bench_fused_add_rmsnorm(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    for (dtype, name) in DTYPES {
        let input = randn((256, 4096), dtype, &device);
        let residual = input.zeros_like().unwrap();
        let weight = randn((1, 4096), dtype, &device)
            .squeeze(0)
            .unwrap();
        let out = input.zeros_like().unwrap();
        let residual_out = input.zeros_like().unwrap();

        c.bench_function(&format!("ayaka_fused_add_rmsnorm_{name}_256x4096"), |b| {
            b.iter(|| {
                fused_add_rmsnorm(&out, &residual_out, &input, &residual, &weight, 1e-5).unwrap();
                device.synchronize().unwrap();
            })
        });
    }
}

fn bench_rope(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let (tokens, heads, kv_heads, head_dim) = (256, 32, 8, 128);

    let positions = Tensor::from_iter((0..tokens).map(|t| (t % 4096) as i64), &device).unwrap();
    let cache = cos_sin_cache(4096, head_dim, 10_000.0, &device).unwrap();

    for (dtype, name) in DTYPES {
        let q = Tensor::randn(0f32, 1f32, (tokens, heads, head_dim), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let k = Tensor::randn(0f32, 1f32, (tokens, kv_heads, head_dim), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        c.bench_function(&format!("ayaka_rope_neox_{name}_256x32x128"), |b| {
            b.iter(|| {
                rope(&q, &k, &positions, &cache, RopeLayout::Neox).unwrap();
                device.synchronize().unwrap();
            })
        });
    }
}

fn bench_silu_and_mul(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    for (dtype, name) in DTYPES {
        let input = randn((256, 2 * 11008), dtype, &device);
        let out = Tensor::zeros((256, 11008), dtype, &device).unwrap();

        c.bench_function(&format!("ayaka_silu_and_mul_{name}_256x11008"), |b| {
            b.iter(|| {
                silu_and_mul(&out, &input).unwrap();
                device.synchronize().unwrap();
            })
        });
    }
}

criterion_group!(
    benches,
    bench_rmsnorm,
    bench_fused_add_rmsnorm,
    bench_rope,
    bench_silu_and_mul
);
criterion_main!(benches);
