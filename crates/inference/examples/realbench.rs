//! Speed and correctness of every installed model on a real GPU.
//!
//! cargo run --release -p clearframe-inference --features gpu --example realbench -- <weights.safetensors>...
//!
//! For each model: the GPU result is compared with the CPU on a small frame, then the GPU (tiled, as the app runs it)
//! is timed at 640x360, 1280x720 and 1920x1080. Prints one JSON line per measurement.
use std::time::Instant;

use clearframe_inference::{Architecture, Device, ImageModel, ImageParams, InterpModel, Precision, RgbImage, Weights};

fn frame(w: usize, h: usize, phase: f32) -> RgbImage {
    let data = (0..w * h * 3)
        .map(|i| {
            let (p, c) = (i / 3, i % 3);
            let (x, y) = ((p % w) as f32 + phase, (p / w) as f32);
            0.5 + 0.35 * ((x * 0.13 + c as f32).sin() * (y * 0.07).cos())
        })
        .collect();
    RgbImage::from_data(w, h, data)
}

fn max_diff(a: &RgbImage, b: &RgbImage) -> f32 {
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn main() {
    for a in clearframe_inference::gpu::list_adapters() {
        println!("{}", serde_json::json!({"adapter": a.index, "name": a.name, "vendor": a.vendor, "kind": a.kind, "api": a.api, "max_buffer_mb": a.max_buffer_mb}));
    }
    let gpu = clearframe_inference::gpu::context(0).expect("no GPU");
    let sizes = [(640usize, 360usize), (1280, 720), (1920, 1080)];
    for path in std::env::args().skip(1) {
        let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().to_string();
        let w = Weights::from_safetensors(std::path::Path::new(&path), &Device::Cpu).unwrap();
        let arch = Architecture::detect(&w);
        if arch == Some(Architecture::Rife) {
            let cpu = InterpModel::load(&w, Precision::Fp32).unwrap();
            let mut g = InterpModel::load(&w, Precision::Fp32).unwrap();
            g.use_gpu(gpu.clone()).unwrap();
            let (a, b) = (frame(256, 144, 0.0), frame(256, 144, 3.0));
            let d = max_diff(&cpu.interpolate(&a, &b, 0.5, 1.0).unwrap(), &g.interpolate(&a, &b, 0.5, 1.0).unwrap());
            println!("{}", serde_json::json!({"model": name, "arch": "rife", "check": "cpu_vs_gpu", "size": "256x144", "max_abs_diff": d}));
            for (sw, sh) in sizes {
                let (a, b) = (frame(sw, sh, 0.0), frame(sw, sh, 3.0));
                let _ = g.interpolate(&a, &b, 0.5, 1.0).unwrap();
                let n = 5;
                let t = Instant::now();
                for _ in 0..n {
                    let _ = g.interpolate(&a, &b, 0.5, 1.0).unwrap();
                }
                let s = t.elapsed().as_secs_f64() / n as f64;
                println!("{}", serde_json::json!({"model": name, "arch": "rife", "size": format!("{sw}x{sh}"), "gpu_s_per_frame": s, "fps": 1.0 / s}));
            }
            continue;
        }
        let Some(arch) = arch.filter(|a| !matches!(a, Architecture::RetinaFace | Architecture::Gfpgan)) else {
            println!("{}", serde_json::json!({"model": name, "skipped": "not a single-network model"}));
            continue;
        };
        let p = ImageParams { noise_sigma: 15.0 / 255.0 };
        let cpu = ImageModel::load(&w, Precision::Fp32).unwrap();
        let mut g = ImageModel::load(&w, Precision::Fp32).unwrap();
        g.use_gpu(gpu.clone()).unwrap();
        let small = frame(160, 90, 0.0);
        let d = max_diff(&cpu.run(&small, &p).unwrap(), &g.run(&small, &p).unwrap());
        let t = Instant::now();
        let _ = cpu.process(&frame(320, 180, 0.0), &p).unwrap();
        let cpu_s = t.elapsed().as_secs_f64();
        println!("{}", serde_json::json!({"model": name, "arch": format!("{arch:?}"), "check": "cpu_vs_gpu", "size": "160x90", "max_abs_diff": d, "cpu_s_320x180": cpu_s, "gpu_tile": g.tile}));
        for (sw, sh) in sizes {
            let img = frame(sw, sh, 0.0);
            let _ = g.process(&img, &p).unwrap();
            let n = 3;
            let t = Instant::now();
            for _ in 0..n {
                let _ = g.process(&img, &p).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / n as f64;
            println!("{}", serde_json::json!({"model": name, "arch": format!("{arch:?}"), "size": format!("{sw}x{sh}"), "gpu_s_per_frame": s, "fps": 1.0 / s}));
        }
    }
}
