//! Compare ClearFrame model adapters with PyTorch reference outputs.
//! usage: parity <refs_dir> <safetensors_dir>
use std::path::Path;

use candle_core::Tensor;
use clearframe_inference::arch::{drunet::DruNet, ffdnet::FfdNet, gfpgan::Gfpgan, nafnet::NafNet, retinaface::RetinaFace, rife::Rife, rrdb::RrdbNet, srvgg::SrVggCompact};
use clearframe_inference::{Device, Weights};

fn read(dir: &Path, name: &str) -> (Vec<Tensor>, Vec<f32>, serde_json::Value) {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap()).unwrap();
    let mut inputs = Vec::new();
    for (i, shape) in meta["inputs"].as_array().unwrap().iter().enumerate() {
        let shape: Vec<usize> = shape.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let bytes = std::fs::read(dir.join(format!("{name}.in{i}.bin"))).unwrap();
        let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        inputs.push(Tensor::from_vec(data, shape.as_slice(), &Device::Cpu).unwrap());
    }
    let bytes = std::fs::read(dir.join(format!("{name}.out.bin"))).unwrap();
    let out = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    (inputs, out, meta)
}

fn cmp(name: &str, got: &Tensor, want: &[f32], tol: f32) -> bool {
    let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(g.len(), want.len(), "{name}: size mismatch");
    let mut max = 0f32;
    let mut mean = 0f64;
    for (a, b) in g.iter().zip(want) {
        let d = (a - b).abs();
        max = max.max(d);
        mean += d as f64;
    }
    mean /= g.len() as f64;
    let ok = max <= tol;
    println!("{:<18} max_abs_err={:.2e} mean_abs_err={:.2e} {}", name, max, mean, if ok { "OK" } else { "FAIL" });
    ok
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let refs = Path::new(&a[1]);
    let st = Path::new(&a[2]);
    let only = a.get(3).cloned();
    let w = |f: &str| Weights::from_safetensors(&st.join(f), &Device::Cpu).unwrap();
    let mut all = true;
    let run = |n: &str| only.as_deref().map_or(true, |o| n.starts_with(o));
    for (name, file) in [("animevideov3", "realesr-animevideov3"), ("general_x4v3", "realesr-general-x4v3")] {
        if !run(name) {
            continue;
        }
        let (i, o, _) = read(refs, name);
        let m = SrVggCompact::load(&w(&format!("{file}.safetensors"))).unwrap();
        all &= cmp(name, &m.forward(&i[0]).unwrap(), &o, 1e-3);
    }
    for (name, file) in [("x4plus", "RealESRGAN_x4plus"), ("anime6b", "RealESRGAN_x4plus_anime_6B"), ("x2plus", "RealESRGAN_x2plus")] {
        if !run(name) {
            continue;
        }
        let (i, o, _) = read(refs, name);
        let m = RrdbNet::load(&w(&format!("{file}.safetensors"))).unwrap();
        all &= cmp(name, &m.forward(&i[0]).unwrap(), &o, 2e-3);
    }
    if run("ffdnet") {
        let (i, o, meta) = read(refs, "ffdnet");
        let m = FfdNet::load(&w("ffdnet_color.safetensors")).unwrap();
        all &= cmp("ffdnet", &m.forward(&i[0], meta["meta"]["sigma"].as_f64().unwrap() as f32).unwrap(), &o, 1e-3);
    }
    if run("drunet") {
        let (i, o, meta) = read(refs, "drunet");
        let m = DruNet::load(&w("drunet_color.safetensors")).unwrap();
        all &= cmp("drunet", &m.forward(&i[0], meta["meta"]["sigma"].as_f64().unwrap() as f32).unwrap(), &o, 1e-3);
    }
    if run("nafnet") {
        let (i, o, _) = read(refs, "nafnet");
        let m = NafNet::load(&w("NAFNet-GoPro-width32.safetensors")).unwrap();
        all &= cmp("nafnet", &m.forward(&i[0]).unwrap(), &o, 1e-3);
    }
    if run("rife") {
        let (i, o, _) = read(refs, "rife");
        let m = Rife::load(&w("rife426.safetensors")).unwrap();
        all &= cmp("rife", &m.forward(&i[0], &i[1], 0.5, 1.0).unwrap(), &o, 2e-3);
    }
    if run("gfpgan") {
        let (i, o, _) = read(refs, "gfpgan");
        let m = Gfpgan::load(&w("GFPGANv1.4.safetensors")).unwrap();
        all &= cmp("gfpgan", &m.forward(&i[0]).unwrap(), &o, 5e-3);
    }
    for (name, file) in [("retina_r50", "detection_Resnet50_Final"), ("retina_mnet", "detection_mobilenet0.25_Final")] {
        if !run(name) {
            continue;
        }
        let m = RetinaFace::load(&w(&format!("{file}.safetensors"))).unwrap();
        let (i, lo, _) = read(refs, &format!("{name}_loc"));
        let (_, co, _) = read(refs, &format!("{name}_conf"));
        let (_, la, _) = read(refs, &format!("{name}_landm"));
        let x = &i[0];
        let bgr = Tensor::cat(&[x.narrow(1, 2, 1).unwrap(), x.narrow(1, 1, 1).unwrap(), x.narrow(1, 0, 1).unwrap()], 1).unwrap();
        let mean = Tensor::new(&[104f32, 117., 123.], &Device::Cpu).unwrap().reshape((1, 3, 1, 1)).unwrap();
        let inp = (bgr * 255.0).unwrap().broadcast_sub(&mean).unwrap();
        let (l, c, lm) = m.forward(&inp).unwrap();
        all &= cmp(&format!("{name}_loc"), &l, &lo, 2e-3);
        all &= cmp(&format!("{name}_conf"), &c, &co, 2e-3);
        all &= cmp(&format!("{name}_landm"), &lm, &la, 2e-3);
    }
    if !all {
        std::process::exit(1);
    }
    println!("ALL PARITY CHECKS PASSED");
}
