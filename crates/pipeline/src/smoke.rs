//! Post-install smoke tests: load converted weights with the real adapter and
//! run a tiny inference, checking output shape and finiteness.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use clearframe_inference::arch::{gfpgan::Gfpgan, retinaface::RetinaFace};
use clearframe_inference::{Architecture, Device, ImageModel, ImageParams, InterpModel, Precision, RgbImage, Weights};

fn pattern(w: usize, h: usize, phase: f32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                img.set(x, y, c, (0.5 + 0.4 * ((x as f32 * 0.37 + y as f32 * 0.21 + c as f32 + phase).sin())).clamp(0.0, 1.0));
            }
        }
    }
    img
}

fn finite(img: &RgbImage) -> bool {
    img.data.iter().all(|v| v.is_finite())
}

fn load(files: &BTreeMap<String, PathBuf>, role: &str) -> Result<Weights, String> {
    let p = files.get(role).ok_or_else(|| format!("missing `{role}` weights"))?;
    Weights::from_safetensors(p, &Device::Cpu).map_err(|e| e.to_string())
}

/// Returns "<architecture>: <description>" on success.
pub fn smoke_test(arch: &str, files: &BTreeMap<String, PathBuf>) -> Result<String, String> {
    let t = Instant::now();
    match arch {
        "rife" => {
            let m = InterpModel::load(&load(files, "weights")?, Precision::Fp32).map_err(|e| e.to_string())?;
            let a = pattern(64, 64, 0.0);
            let b = pattern(64, 64, 0.6);
            let out = m.interpolate(&a, &b, 0.5, 1.0).map_err(|e| e.to_string())?;
            if (out.width, out.height) != (64, 64) || !finite(&out) {
                return Err("unexpected interpolation output".into());
            }
            Ok(format!("rife: 64x64 pair interpolated in {:.2}s", t.elapsed().as_secs_f64()))
        }
        "gfpgan" => {
            let det = RetinaFace::load(&load(files, "detector")?).map_err(|e| e.to_string())?;
            let face = pattern(96, 96, 0.3).to_tensor(&Device::Cpu, clearframe_inference::DType::F32).map_err(|e| e.to_string())?;
            det.detect(&face, 0.97, 0.4).map_err(|e| e.to_string())?;
            let g = Gfpgan::load(&load(files, "restorer")?).map_err(|e| e.to_string())?;
            let x = pattern(512, 512, 0.1).to_tensor(&Device::Cpu, clearframe_inference::DType::F32).map_err(|e| e.to_string())?;
            let y = RgbImage::from_tensor(&g.forward(&x).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            if (y.width, y.height) != (512, 512) || !finite(&y) {
                return Err("unexpected face restoration output".into());
            }
            Ok(format!("gfpgan: detector and 512x512 restoration ran in {:.2}s", t.elapsed().as_secs_f64()))
        }
        _ => {
            let w = load(files, "weights")?;
            let detected = Architecture::detect(&w).ok_or("unrecognised network architecture")?;
            if arch != "auto" && detected.id() != arch {
                return Err(format!("expected {arch} but the weights look like {}", detected.id()));
            }
            if !matches!(detected, Architecture::SrVggCompact | Architecture::RrdbNet | Architecture::FfdNet | Architecture::DruNet | Architecture::NafNet) {
                return Err(format!("{} cannot be used as an image model", detected.id()));
            }
            let mut m = ImageModel::load(&w, Precision::Fp32).map_err(|e| e.to_string())?;
            m.tile = 0;
            let img = pattern(32, 24, 0.0);
            let out = m.process(&img, &ImageParams::default()).map_err(|e| e.to_string())?;
            let s = m.scale();
            if (out.width, out.height) != (32 * s, 24 * s) || !finite(&out) {
                return Err("unexpected output size or values".into());
            }
            Ok(format!("{}: x{s}, 32x24 -> {}x{} in {:.2}s", detected.id(), out.width, out.height, t.elapsed().as_secs_f64()))
        }
    }
}
