//! RMSNorm kernel benchmark (requires `--features cuda` and a CUDA device).
//!
//! Run with: `cargo bench -p ayaka-candle --features cuda`

use ayaka_candle::ops::rmsnorm;
use candle_core::{DType, Device, Tensor};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_rmsnorm(c: &mut Criterion) {
    let device = Device::new_cuda(0).expect("CUDA device 0");

    for (dtype, name) in [(DType::F32, "f32"), (DType::F16, "f16")] {
        let input = Tensor::randn(0f32, 1f32, (256, 4096), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let weight = Tensor::randn(0f32, 1f32, 4096, &device)
            .unwrap()
            .to_dtype(dtype)
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

criterion_group!(benches, bench_rmsnorm);
criterion_main!(benches);
