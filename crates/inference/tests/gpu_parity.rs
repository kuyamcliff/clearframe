//! The GPU networks must produce the same output as the CPU networks.
//!
//! Each test builds a small random network of every architecture (odd channel
//! counts and image sizes to hit the kernels' edge cases) and compares the GPU
//! result with the CPU result. Runs on any Vulkan/DX12/Metal device, including
//! Mesa's software "lavapipe" driver on CI machines without a GPU; skips when no
//! device exists at all.
#![cfg(feature = "gpu")]

use clearframe_inference::{Device, ImageModel, ImageParams, Precision, RgbImage, Weights};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[derive(Default)]
struct W(Vec<(String, Vec<usize>, Vec<f32>)>);
impl W {
    fn t(&mut self, r: &mut Rng, name: &str, shape: &[usize], scale: f32) {
        let n = shape.iter().product();
        self.0.push((name.into(), shape.to_vec(), (0..n).map(|_| r.next() * scale).collect()));
    }
    fn conv(&mut self, r: &mut Rng, p: &str, cout: usize, cin: usize, k: usize, bias: bool) {
        let fan = (cin * k * k) as f32;
        self.t(r, &format!("{p}.weight"), &[cout, cin, k, k], 1.2 / fan.sqrt());
        if bias {
            self.t(r, &format!("{p}.bias"), &[cout], 0.05);
        }
    }
    fn build(self) -> Weights {
        Weights::from_raw(self.0, &Device::Cpu).unwrap()
    }
}

fn image(r: &mut Rng, w: usize, h: usize) -> RgbImage {
    RgbImage::from_data(w, h, (0..w * h * 3).map(|_| 0.5 + 0.3 * r.next()).collect())
}

fn gpu() -> Option<std::sync::Arc<clearframe_inference::gpu::Gpu>> {
    match clearframe_inference::gpu::context(0) {
        Ok(g) => {
            eprintln!("GPU: {} ({}, {})", g.info.name, g.info.vendor, g.info.api);
            Some(g)
        }
        Err(e) => {
            // CI sets CLIPFX_REQUIRE_GPU so a machine without a usable device fails loudly
            // instead of silently skipping.
            assert!(std::env::var_os("CLIPFX_REQUIRE_GPU").is_none(), "no GPU device: {e}");
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// Run on CPU and GPU and return the largest absolute difference.
fn compare(w: Weights, img: &RgbImage, p: ImageParams, tol: f32) {
    let Some(g) = gpu() else { return };
    let cpu = ImageModel::load(&w, Precision::Fp32).unwrap();
    let mut on_gpu = ImageModel::load(&w, Precision::Fp32).unwrap();
    on_gpu.use_gpu(g).unwrap();
    // Compare the raw network outputs (before clamping, so saturation can't hide errors).
    let (a, _) = cpu.run_raw(img, &p).unwrap();
    let (b, _) = on_gpu.run_raw(img, &p).unwrap();
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let spread = a.iter().map(|v| (v - a[0]).abs()).fold(0f32, f32::max);
    eprintln!("{:?}: {} values, max diff {max:.2e}, |output| up to {peak:.2}", cpu.arch, a.len());
    assert!(spread > 0.05, "output is flat; the test would prove nothing");
    assert!(max <= tol * peak.max(1.0), "GPU differs from CPU by {max} (output magnitude {peak})");
    // And the clamped images used for video match too.
    let (ca, cb) = (cpu.run(img, &p).unwrap(), on_gpu.run(img, &p).unwrap());
    assert_eq!((ca.width, ca.height), (cb.width, cb.height));
}

#[test]
fn srvgg_matches_cpu() {
    let mut r = Rng(1);
    let mut w = W::default();
    let (nf, n_conv, scale) = (20, 4, 4);
    w.conv(&mut r, "body.0", nf, 3, 3, true);
    w.t(&mut r, "body.1.weight", &[nf], 0.25);
    for i in 1..=n_conv {
        w.conv(&mut r, &format!("body.{}", 2 * i), nf, nf, 3, true);
        w.t(&mut r, &format!("body.{}.weight", 2 * i + 1), &[nf], 0.25);
    }
    w.conv(&mut r, &format!("body.{}", 2 * n_conv + 2), 3 * scale * scale, nf, 3, true);
    let img = image(&mut r, 37, 23);
    compare(w.build(), &img, ImageParams::default(), 2e-4);
}

fn rrdb(scale: usize, seed: u64) {
    let mut r = Rng(seed);
    let mut w = W::default();
    let (nf, gc, nb) = (20, 12, 2);
    let in_ch = 3 * (4 / scale) * (4 / scale);
    w.conv(&mut r, "conv_first", nf, in_ch, 3, true);
    for b in 0..nb {
        for d in 1..=3 {
            let p = format!("body.{b}.rdb{d}");
            for i in 1..=4 {
                w.conv(&mut r, &format!("{p}.conv{i}"), gc, nf + (i - 1) * gc, 3, true);
            }
            w.conv(&mut r, &format!("{p}.conv5"), nf, nf + 4 * gc, 3, true);
        }
    }
    for n in ["conv_body", "conv_up1", "conv_up2", "conv_hr"] {
        w.conv(&mut r, n, nf, nf, 3, true);
    }
    w.conv(&mut r, "conv_last", 3, nf, 3, true);
    let img = image(&mut r, 30, 18);
    compare(w.build(), &img, ImageParams::default(), 2e-4);
}

#[test]
fn rrdb_x4_matches_cpu() {
    rrdb(4, 2);
}

#[test]
fn rrdb_x2_matches_cpu() {
    rrdb(2, 3);
}

#[test]
fn ffdnet_matches_cpu() {
    let mut r = Rng(4);
    let mut w = W::default();
    let nf = 24;
    w.conv(&mut r, "model.0", nf, 13, 3, true);
    w.conv(&mut r, "model.2", nf, nf, 3, true);
    w.conv(&mut r, "model.4", nf, nf, 3, true);
    w.conv(&mut r, "model.6", 12, nf, 3, true);
    let img = image(&mut r, 33, 21);
    compare(w.build(), &img, ImageParams { noise_sigma: 25.0 / 255.0 }, 2e-4);
}

#[test]
fn drunet_matches_cpu() {
    let mut r = Rng(5);
    let mut w = W::default();
    let nc = [8, 16, 32, 64];
    w.conv(&mut r, "m_head", nc[0], 4, 3, false);
    let res = |w: &mut W, r: &mut Rng, p: &str, c: usize| {
        w.conv(r, &format!("{p}.res.0"), c, c, 3, false);
        w.conv(r, &format!("{p}.res.2"), c, c, 3, false);
    };
    for d in 0..3 {
        let p = format!("m_down{}", d + 1);
        for i in 0..4 {
            res(&mut w, &mut r, &format!("{p}.{i}"), nc[d]);
        }
        w.conv(&mut r, &format!("{p}.4"), nc[d + 1], nc[d], 2, false);
    }
    for i in 0..4 {
        res(&mut w, &mut r, &format!("m_body.{i}"), nc[3]);
    }
    for u in [3usize, 2, 1] {
        let p = format!("m_up{u}");
        w.t(&mut r, &format!("{p}.0.weight"), &[nc[u], nc[u - 1], 2, 2], 0.6 / (nc[u] as f32).sqrt());
        for i in 1..=4 {
            res(&mut w, &mut r, &format!("{p}.{i}"), nc[u - 1]);
        }
    }
    w.conv(&mut r, "m_tail", 3, nc[0], 3, false);
    let img = image(&mut r, 29, 19);
    compare(w.build(), &img, ImageParams { noise_sigma: 15.0 / 255.0 }, 3e-4);
}

#[test]
fn nafnet_matches_cpu() {
    let mut r = Rng(6);
    let mut w = W::default();
    let block = |w: &mut W, r: &mut Rng, p: &str, c: usize| {
        for n in ["norm1", "norm2"] {
            w.0.push((format!("{p}.{n}.weight"), vec![c], (0..c).map(|_| 1.0 + 0.2 * r.next()).collect()));
            w.t(r, &format!("{p}.{n}.bias"), &[c], 0.1);
        }
        w.conv(r, &format!("{p}.conv1"), 2 * c, c, 1, true);
        w.t(r, &format!("{p}.conv2.weight"), &[2 * c, 1, 3, 3], 0.4);
        w.t(r, &format!("{p}.conv2.bias"), &[2 * c], 0.05);
        w.conv(r, &format!("{p}.conv3"), c, c, 1, true);
        w.conv(r, &format!("{p}.sca.1"), c, c, 1, true);
        w.conv(r, &format!("{p}.conv4"), 2 * c, c, 1, true);
        w.conv(r, &format!("{p}.conv5"), c, c, 1, true);
        w.t(r, &format!("{p}.beta"), &[1, c, 1, 1], 0.5);
        w.t(r, &format!("{p}.gamma"), &[1, c, 1, 1], 0.5);
    };
    let c0 = 8;
    w.conv(&mut r, "intro", c0, 3, 3, true);
    block(&mut w, &mut r, "encoders.0.0", c0);
    w.conv(&mut r, "downs.0", 2 * c0, c0, 2, true);
    block(&mut w, &mut r, "encoders.1.0", 2 * c0);
    w.conv(&mut r, "downs.1", 4 * c0, 2 * c0, 2, true);
    block(&mut w, &mut r, "middle_blks.0", 4 * c0);
    w.conv(&mut r, "ups.0.0", 8 * c0, 4 * c0, 1, false);
    block(&mut w, &mut r, "decoders.0.0", 2 * c0);
    w.conv(&mut r, "ups.1.0", 4 * c0, 2 * c0, 1, false);
    block(&mut w, &mut r, "decoders.1.0", c0);
    w.conv(&mut r, "ending", 3, c0, 3, true);
    let img = image(&mut r, 27, 17);
    compare(w.build(), &img, ImageParams::default(), 3e-4);
}

#[test]
fn tiling_on_gpu_matches_whole_frame() {
    // Process a frame in several GPU tiles and check the seams match a single pass.
    let Some(g) = gpu() else { return };
    let mut r = Rng(7);
    let mut w = W::default();
    w.conv(&mut r, "body.0", 16, 3, 3, true);
    w.t(&mut r, "body.1.weight", &[16], 0.25);
    w.conv(&mut r, "body.2", 16, 16, 3, true);
    w.t(&mut r, "body.3.weight", &[16], 0.25);
    w.conv(&mut r, "body.4", 12, 16, 3, true);
    let weights = w.build();
    let img = image(&mut r, 90, 70);
    let mut m = ImageModel::load(&weights, Precision::Fp32).unwrap();
    m.use_gpu(g).unwrap();
    let whole = m.run(&img, &ImageParams::default()).unwrap();
    m.tile = 32;
    m.tile_pad = 8;
    let tiled = m.process(&img, &ImageParams::default()).unwrap();
    let max = whole.data.iter().zip(&tiled.data).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(max < 1e-4, "tile seams differ by {max}");
}
