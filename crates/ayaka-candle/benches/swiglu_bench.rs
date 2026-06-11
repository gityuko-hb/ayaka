//! silu_and_mul kernel benchmark (requires `--features cuda`).
//!
//! Run with: `cargo bench -p ayaka-candle --features cuda`

use ayaka_candle::ops::silu_and_mul;
use candle_core::{DType, Device, Tensor};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_silu_and_mul(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");

    for (dtype, name) in [(DType::F32, "f32"), (DType::F16, "f16")] {
        let input = Tensor::randn(0f32, 1f32, (256, 2 * 11008), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let out = Tensor::zeros((256, 11008), dtype, &device).unwrap();

        c.bench_function(&format!("ayaka_silu_and_mul_{name}_256x11008"), |b| {
            b.iter(|| {
                silu_and_mul(&out, &input).unwrap();
                device.synchronize().unwrap();
            })
        });
    }
}

criterion_group!(benches, bench_silu_and_mul);
criterion_main!(benches);
