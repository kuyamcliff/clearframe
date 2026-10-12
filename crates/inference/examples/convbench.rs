//! Micro-benchmark of the 3x3 convolution that dominates SR models.
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let (c, h, w) = (64usize, 360usize, 640usize);
    let x = Tensor::randn(0f32, 1.0, (1, c, h, w), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 0.05, (c, c, 3, 3), &Device::Cpu)?;
    for _ in 0..2 {
        let t = Instant::now();
        let y = x.conv2d(&k, 1, 1, 1, 1)?;
        let _ = y.sum_all()?.to_scalar::<f32>()?;
        let s = t.elapsed().as_secs_f64();
        let gflop = (c * c * 9 * h * w * 2) as f64 / 1e9;
        println!("candle conv2d {c}x{c} 3x3 @ {w}x{h}: {:.3}s  {:.1} GFLOP/s", s, gflop / s);
    }
    // Same work as one GEMM (im2col excluded).
    let a = Tensor::randn(0f32, 1.0, (h * w, c * 9), &Device::Cpu)?;
    let b = Tensor::randn(0f32, 1.0, (c * 9, c), &Device::Cpu)?;
    for _ in 0..2 {
        let t = Instant::now();
        let y = a.matmul(&b)?;
        let _ = y.sum_all()?.to_scalar::<f32>()?;
        let s = t.elapsed().as_secs_f64();
        println!("gemm {}x{}x{}: {:.3}s  {:.1} GFLOP/s", h * w, c * 9, c, s, (h * w * c * 9 * c * 2) as f64 / 1e9 / s);
    }
    let _ = DType::F32;
    Ok(())
}
