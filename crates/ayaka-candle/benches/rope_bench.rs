//! rope kernel benchmark (requires `--features cuda`).
//!
//! Run with: `cargo bench -p ayaka-candle --features cuda`

use ayaka_candle::ops::{RopeLayout, cos_sin_cache, rope};
use candle_core::{DType, Device, Tensor};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_rope(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");
    let (tokens, heads, kv_heads, head_dim) = (256, 32, 8, 128);

    let positions = Tensor::from_iter((0..tokens).map(|t| (t % 4096) as i64), &device).unwrap();
    let cache = cos_sin_cache(4096, head_dim, 10_000.0, &device).unwrap();

    for (dtype, name) in [(DType::F32, "f32"), (DType::F16, "f16")] {
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

criterion_group!(benches, bench_rope);
criterion_main!(benches);
