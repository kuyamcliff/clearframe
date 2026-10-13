//! Compare CPU and GPU output and speed for a real model.
//!
//! cargo run --release -p clearframe-inference --features gpu --example gpubench -- <model.safetensors> [width height]
use std::time::Instant;

use clearframe_inference::{Device, ImageModel, ImageParams, Precision, RgbImage, Weights};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: gpubench <model.safetensors> [width height]");
    let w: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(320);
    let h: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(180);
    for a in clearframe_inference::gpu::list_adapters() {
        println!("adapter {}: {} [{}, {}, {}] max buffer {} MB", a.index, a.name, a.vendor, a.kind, a.api, a.max_buffer_mb);
    }
    let weights = Weights::from_safetensors(std::path::Path::new(path), &Device::Cpu).unwrap();
    // A smooth gradient with some detail so the model has something to do.
    let data = (0..w * h * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (x, y) = ((p % w) as f32, (p / w) as f32);
            0.5 + 0.35 * ((x * 0.13 + c as f32).sin() * (y * 0.07).cos())
        })
        .collect();
    let img = RgbImage::from_data(w, h, data);
    let p = ImageParams::default();
    let cpu = ImageModel::load(&weights, Precision::Fp32).unwrap();
    let mut gpu = ImageModel::load(&weights, Precision::Fp32).unwrap();
    gpu.use_gpu(clearframe_inference::gpu::context(0).unwrap()).unwrap();
    let _ = gpu.run(&img, &p).unwrap(); // warm-up (shader compilation)
    let t = Instant::now();
    let (a, _) = cpu.run_raw(&img, &p).unwrap();
    let cpu_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (b, _) = gpu.run_raw(&img, &p).unwrap();
    let gpu_s = t.elapsed().as_secs_f64();
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("{:?} {w}x{h}: CPU {cpu_s:.3}s, GPU {gpu_s:.3}s ({}), max |CPU-GPU| {max:.2e}", cpu.arch, gpu.gpu_name().unwrap());
    // Real models produce values around 0..1; anything beyond float rounding is a bug.
    if max > 1e-3 {
        eprintln!("GPU output differs from the CPU by {max}");
        std::process::exit(1);
    }
}
