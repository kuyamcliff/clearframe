//! Renders with real AI models. Requires converted weights:
//! `CLEARFRAME_TEST_MODELS=/path/to/safetensors cargo test --release -p clearframe-pipeline --test models_e2e`
//! Expected files: realesr-animevideov3, realesr-general-x4v3, realesr-general-wdn-x4v3,
//! ffdnet_color, NAFNet-GoPro-width32, rife426, GFPGANv1.4, detection_mobilenet0.25_Final (.safetensors).

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use clearframe_inference::{Device, ImageModel, ImageParams, Precision, RgbImage, Weights};
use clearframe_media::probe::probe;
use clearframe_pipeline::spec::*;
use common::*;

fn models_dir() -> Option<PathBuf> {
    std::env::var("CLEARFRAME_TEST_MODELS").ok().map(PathBuf::from).filter(|p| p.is_dir())
}

fn resolved(dir: &PathBuf, id: &str, files: &[(&str, &str)]) -> (String, ResolvedModel) {
    let mut f = BTreeMap::new();
    let mut h = BTreeMap::new();
    for (role, name) in files {
        let p = dir.join(format!("{name}.safetensors"));
        h.insert(role.to_string(), sha(&p));
        f.insert(role.to_string(), p);
    }
    let scale = if id.contains("realesr") { 4 } else { 1 };
    (id.to_string(), ResolvedModel { id: id.into(), version: "test".into(), files: f, sha256: h, architecture: String::new(), scale, memory_per_megapixel_mb: 1500.0 })
}

macro_rules! setup {
    () => {{
        let (Some(t), Some(m)) = (tools(), models_dir()) else {
            eprintln!("CLEARFRAME_TEST_MODELS not set; skipping");
            return;
        };
        (t, m)
    }};
}

#[test]
fn tiled_matches_untiled() {
    let Some(m) = models_dir() else { return };
    let w = Weights::from_safetensors(&m.join("realesr-animevideov3.safetensors"), &Device::Cpu).unwrap();
    let mut model = ImageModel::load(&w, Precision::Fp32).unwrap();
    let mut img = RgbImage::new(96, 72);
    for (i, v) in img.data.iter_mut().enumerate() {
        *v = ((i * 37 % 251) as f32 / 251.0) * 0.6 + 0.2;
    }
    model.tile = 0;
    let whole = model.process(&img, &ImageParams::default()).unwrap();
    model.tile = 40;
    model.tile_pad = 16;
    let tiled = model.process(&img, &ImageParams::default()).unwrap();
    let max = whole.data.iter().zip(&tiled.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // The compact network's receptive field is 35 px; 16 px padding leaves tiny differences only.
    assert!(max < 0.02, "max tile seam difference {max}");
}

#[test]
fn enhance_denoise_deblur_face_interpolate() {
    let (t, m) = setup!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "scale=160:120", 0.5);
    let mut models = BTreeMap::new();
    for (id, files) in [
        ("xinntao.realesr-animevideo-v3", vec![("weights", "realesr-animevideov3")]),
        ("xinntao.realesr-general-v3", vec![("weights", "realesr-general-x4v3"), ("wdn", "realesr-general-wdn-x4v3")]),
        ("kair.ffdnet-color", vec![("weights", "ffdnet_color")]),
        ("megvii.nafnet-gopro-w32", vec![("weights", "NAFNet-GoPro-width32")]),
        ("hzwer.rife-v4.26", vec![("weights", "rife426")]),
        ("tencentarc.gfpgan-v1.4", vec![("restorer", "GFPGANv1.4"), ("detector", "detection_mobilenet0.25_Final")]),
    ] {
        let (k, v) = resolved(&m, id, &files);
        models.insert(k, v);
    }
    let p = Pipeline {
        stages: vec![
            Stage::new(Operation::Denoise(DenoiseSettings { engine: DenoiseEngine::Ffdnet, strength: 10.0, ..Default::default() })),
            Stage::new(Operation::Deblur(DeblurSettings { strength: 0.5, ..Default::default() })),
            Stage::new(Operation::Enhance(EnhanceSettings { model_id: "xinntao.realesr-general-v3".into(), denoise: Some(0.5), target: SizeTarget::Height { height: 360 }, ..Default::default() })),
            Stage::new(Operation::FaceRestore(FaceSettings::default())),
            Stage::new(Operation::Interpolate(InterpolateSettings::default())),
        ],
    };
    let mut spec = job(d.path(), &src, p, OutputSettings::default(), "ai");
    spec.models = models;
    let r = run(&t, &spec).unwrap();
    let info = probe(&t, &r.output).unwrap();
    let v = info.video.as_ref().unwrap();
    assert_eq!((v.width, v.height), (480, 360));
    assert!((v.avg_frame_rate.as_f64() - 60.0).abs() < 0.2);
    assert_eq!(frame_md5(&r.output).len(), 30);
    assert_eq!(r.models.len(), 6, "all model hashes recorded");
    eprintln!("timings: {:?}", r.timings);
}

#[test]
fn anime_upscale_quality_vs_bicubic() {
    // The model output must be closer to a high-res reference than plain bicubic.
    let (t, m) = setup!();
    let d = tempfile::tempdir().unwrap();
    let hi = d.path().join("hi.mkv");
    ffmpeg(&["-f", "lavfi", "-i", "testsrc2=size=640x480:rate=10", "-t", "0.5", "-c:v", "ffv1", hi.to_str().unwrap()]);
    let lo = d.path().join("lo.mkv");
    ffmpeg(&["-i", hi.to_str().unwrap(), "-vf", "scale=160:120:flags=area", "-c:v", "ffv1", lo.to_str().unwrap()]);
    let bic = d.path().join("bic.mkv");
    ffmpeg(&["-i", lo.to_str().unwrap(), "-vf", "scale=640:480:flags=bicubic", "-c:v", "ffv1", bic.to_str().unwrap()]);
    let (k, v) = resolved(&m, "xinntao.realesr-animevideo-v3", &[("weights", "realesr-animevideov3")]);
    let mut spec = job(d.path(), &lo, Pipeline::single(Operation::Enhance(EnhanceSettings::default())), lossless(), "sr");
    spec.models.insert(k, v);
    let r = run(&t, &spec).unwrap();
    let p_model = psnr(&hi, &r.output);
    let p_bic = psnr(&hi, &bic);
    eprintln!("PSNR model {p_model:.2} dB vs bicubic {p_bic:.2} dB");
    assert!(p_model > p_bic, "model {p_model} should beat bicubic {p_bic} on clean synthetic line art");
}

#[test]
fn smoke_tests_accept_real_and_reject_mismatched_weights() {
    let Some(m) = models_dir() else { return };
    let f = |n: &str| -> std::collections::BTreeMap<String, PathBuf> { [("weights".to_string(), m.join(format!("{n}.safetensors")))].into_iter().collect() };
    let ok = clearframe_pipeline::smoke::smoke_test("srvgg_compact", &f("realesr-animevideov3")).unwrap();
    assert!(ok.contains("x4"), "{ok}");
    assert!(clearframe_pipeline::smoke::smoke_test("auto", &f("ffdnet_color")).unwrap().starts_with("ffdnet"));
    assert!(clearframe_pipeline::smoke::smoke_test("rife", &f("rife426")).is_ok());
    assert!(clearframe_pipeline::smoke::smoke_test("rrdbnet", &f("realesr-animevideov3")).is_err());
    let face: std::collections::BTreeMap<String, PathBuf> = [("restorer".to_string(), m.join("GFPGANv1.4.safetensors")), ("detector".to_string(), m.join("detection_mobilenet0.25_Final.safetensors"))].into_iter().collect();
    assert!(clearframe_pipeline::smoke::smoke_test("gfpgan", &face).is_ok());
}
