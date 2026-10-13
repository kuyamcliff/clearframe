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

#[test]
fn concurrent_forwards_on_one_gpu_match_sequential() {
    // Pipeline stages and frames in flight run forward passes from several threads on the shared GPU
    // context. Pooled buffers are reused between passes, so the results must not depend on interleaving.
    let Some(g) = gpu() else { return };
    let mut r = Rng(11);
    let mut w = W::default();
    w.conv(&mut r, "body.0", 16, 3, 3, true);
    w.t(&mut r, "body.1.weight", &[16], 0.25);
    w.conv(&mut r, "body.2", 16, 16, 3, true);
    w.t(&mut r, "body.3.weight", &[16], 0.25);
    w.conv(&mut r, "body.4", 12, 16, 3, true);
    let weights = w.build();
    let mut m = ImageModel::load(&weights, Precision::Fp32).unwrap();
    m.use_gpu(g).unwrap();
    // Small tiles: many short passes per frame, so the threads interleave often.
    m.tile = 16;
    m.tile_pad = 4;
    let imgs: Vec<RgbImage> = (0..8).map(|i| image(&mut r, 61 + 7 * i, 43 + 5 * i)).collect();
    let p = ImageParams::default();
    let expect: Vec<RgbImage> = imgs.iter().map(|img| m.process(img, &p).unwrap()).collect();
    let m = &m;
    std::thread::scope(|s| {
        for (img, want) in imgs.iter().zip(&expect) {
            let p = &p;
            s.spawn(move || {
                for _ in 0..25 {
                    let got = m.process(img, p).unwrap();
                    let max = got.data.iter().zip(&want.data).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    assert!(max == 0.0, "concurrent result differs by {max}");
                }
            });
        }
    });
}

/// RIFE: same structure as v4.26 (4-channel encoder, five IFBlocks passing 4 flow + 1 mask + 8 feature channels),
/// narrower layers. Weights are kept small so the predicted flow stays in a realistic range.
#[test]
fn rife_on_gpu_matches_cpu() {
    let Some(g) = gpu() else { return };
    let mut r = Rng(7);
    let mut w = W::default();
    w.conv(&mut r, "encode.cnn0", 8, 3, 3, true);
    w.conv(&mut r, "encode.cnn1", 8, 8, 3, true);
    w.conv(&mut r, "encode.cnn2", 8, 8, 3, true);
    w.t(&mut r, "encode.cnn3.weight", &[8, 4, 4, 4], 0.2);
    w.t(&mut r, "encode.cnn3.bias", &[4], 0.05);
    for b in 0..5 {
        let cin = if b == 0 { 15 } else { 28 };
        let p = format!("block{b}");
        w.conv(&mut r, &format!("{p}.conv0.0.0"), 8, cin, 3, true);
        w.conv(&mut r, &format!("{p}.conv0.1.0"), 16, 8, 3, true);
        for i in 0..2 {
            w.conv(&mut r, &format!("{p}.convblock.{i}.conv"), 16, 16, 3, true);
            w.t(&mut r, &format!("{p}.convblock.{i}.beta"), &[1, 16, 1, 1], 0.5);
        }
        w.t(&mut r, &format!("{p}.lastconv.0.weight"), &[16, 52, 4, 4], 0.08);
        w.t(&mut r, &format!("{p}.lastconv.0.bias"), &[52], 0.05);
    }
    let weights = w.build();
    let cpu = clearframe_inference::InterpModel::load(&weights, Precision::Fp32).unwrap();
    let mut on_gpu = clearframe_inference::InterpModel::load(&weights, Precision::Fp32).unwrap();
    on_gpu.use_gpu(g).unwrap();
    assert!(on_gpu.on_gpu());
    // Odd size: exercises the replicate padding to the 64-pixel alignment and the crop back.
    let (a, b) = (image(&mut r, 93, 70), image(&mut r, 93, 70));
    for (t, scale) in [(0.5f32, 1.0f64), (0.25, 1.0), (0.5, 0.5)] {
        let x = cpu.interpolate(&a, &b, t, scale).unwrap();
        let y = on_gpu.interpolate(&a, &b, t, scale).unwrap();
        assert_eq!((x.width, x.height), (y.width, y.height));
        let max = x.data.iter().zip(&y.data).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        let spread = x.data.iter().map(|v| (v - x.data[0]).abs()).fold(0f32, f32::max);
        eprintln!("RIFE t={t} scale={scale}: max diff {max:.2e} (output spread {spread:.2})");
        assert!(spread > 0.05, "output is flat; the test would prove nothing");
        assert!(max <= 2e-3, "GPU RIFE differs from CPU by {max}");
    }
}

/// GFPGAN v1.4 layout (512px, 7 down / 7 up blocks, StyleGAN2 decoder with SFT), narrow channels.
#[test]
fn gfpgan_on_gpu_matches_cpu() {
    let Some(g) = gpu() else { return };
    let mut r = Rng(11);
    let mut w = W::default();
    let (c, nsf) = (8usize, 8usize);
    w.conv(&mut r, "conv_body_first", c, 3, 1, true);
    for i in 0..7 {
        for p in ["conv_body_down", "conv_body_up"] {
            w.conv(&mut r, &format!("{p}.{i}.conv1"), c, c, 3, true);
            w.conv(&mut r, &format!("{p}.{i}.conv2"), c, c, 3, true);
            w.conv(&mut r, &format!("{p}.{i}.skip"), c, c, 1, false);
        }
        for p in ["condition_scale", "condition_shift"] {
            w.conv(&mut r, &format!("{p}.{i}.0"), c, c, 3, true);
            w.conv(&mut r, &format!("{p}.{i}.2"), c / 2, c, 3, true);
        }
    }
    w.conv(&mut r, "final_conv", c, c, 3, true);
    w.t(&mut r, "final_linear.weight", &[16 * nsf, c * 16], 0.1);
    w.t(&mut r, "final_linear.bias", &[16 * nsf], 0.05);
    w.t(&mut r, "stylegan_decoder.constant_input.weight", &[1, c, 4, 4], 0.5);
    let modconv = |w: &mut W, r: &mut Rng, p: &str, out: usize, k: usize| {
        w.t(r, &format!("{p}.modulated_conv.weight"), &[1, out, c, k, k], 1.0 / ((c * k * k) as f32).sqrt());
        w.t(r, &format!("{p}.modulated_conv.modulation.weight"), &[c, nsf], 0.3);
        w.t(r, &format!("{p}.modulated_conv.modulation.bias"), &[c], 0.3);
    };
    modconv(&mut w, &mut r, "stylegan_decoder.style_conv1", c, 3);
    w.t(&mut r, "stylegan_decoder.style_conv1.weight", &[1], 0.1);
    w.t(&mut r, "stylegan_decoder.style_conv1.bias", &[1, c, 1, 1], 0.05);
    modconv(&mut w, &mut r, "stylegan_decoder.to_rgb1", 3, 1);
    w.t(&mut r, "stylegan_decoder.to_rgb1.bias", &[1, 3, 1, 1], 0.05);
    w.t(&mut r, "stylegan_decoder.noises.noise0", &[1, 1, 4, 4], 1.0);
    for j in 0..14 {
        let res = 8usize << (j / 2);
        let p = format!("stylegan_decoder.style_convs.{j}");
        modconv(&mut w, &mut r, &p, c, 3);
        w.t(&mut r, &format!("{p}.weight"), &[1], 0.1);
        w.t(&mut r, &format!("{p}.bias"), &[1, c, 1, 1], 0.05);
        w.t(&mut r, &format!("stylegan_decoder.noises.noise{}", j + 1), &[1, 1, res, res], 1.0);
    }
    for k in 0..7 {
        let p = format!("stylegan_decoder.to_rgbs.{k}");
        modconv(&mut w, &mut r, &p, 3, 1);
        w.t(&mut r, &format!("{p}.bias"), &[1, 3, 1, 1], 0.05);
    }
    let weights = w.build();
    let cpu = clearframe_inference::arch::gfpgan::Gfpgan::load(&weights).unwrap();
    let gnet = clearframe_inference::gpu::nets::GGfpgan::new(&g, &cpu).unwrap();
    let face = image(&mut r, 512, 512).to_tensor(&Device::Cpu, clearframe_inference::DType::F32).unwrap();
    let a = cpu.forward(&face).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let planar = clearframe_inference::gpu::nets::Planar::from_tensor(&face).unwrap();
    let b = gnet.forward(&g, &planar).unwrap().data;
    let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let spread = a.iter().map(|v| (v - a[0]).abs()).fold(0f32, f32::max);
    let mean = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    let big = a.iter().zip(&b).filter(|(x, y)| (*x - *y).abs() > 1e-3).count();
    let (amin, amax) = a.iter().fold((f32::MAX, f32::MIN), |m, v| (m.0.min(*v), m.1.max(*v)));
    eprintln!("GFPGAN: {} values, max diff {max:.2e}, mean diff {mean:.2e}, {big} values off by >1e-3, CPU range {amin:.3}..{amax:.3}", a.len());
    assert!(spread > 0.05, "output is flat; the test would prove nothing");
    assert!(max <= 2e-3, "GPU GFPGAN differs from CPU by {max}");
}

/// RetinaFace ResNet-50 layout (stem, 3/4/6/3 bottlenecks, FPN, SSH, heads) with narrow channels and random BN stats.
#[test]
fn retinaface_on_gpu_matches_cpu() {
    let Some(g) = gpu() else { return };
    let mut r = Rng(13);
    let mut w = W::default();
    let bn = |w: &mut W, r: &mut Rng, p: &str, c: usize| {
        w.t(r, &format!("{p}.weight"), &[c], 0.3);
        w.0.last_mut().unwrap().2.iter_mut().for_each(|v| *v += 1.0);
        w.t(r, &format!("{p}.bias"), &[c], 0.1);
        w.t(r, &format!("{p}.running_mean"), &[c], 0.1);
        w.t(r, &format!("{p}.running_var"), &[c], 0.2);
        w.0.last_mut().unwrap().2.iter_mut().for_each(|v| *v = v.abs() + 0.5);
    };
    w.conv(&mut r, "body.conv1", 8, 3, 7, false);
    bn(&mut w, &mut r, "body.bn1", 8);
    let (mid, out) = (4usize, 16usize);
    let mut cin = 8;
    let mut level_out = Vec::new();
    for (li, n) in [3usize, 4, 6, 3].iter().enumerate() {
        for bi in 0..*n {
            let p = format!("body.layer{}.{bi}", li + 1);
            w.conv(&mut r, &format!("{p}.conv1"), mid, cin, 1, false);
            bn(&mut w, &mut r, &format!("{p}.bn1"), mid);
            w.conv(&mut r, &format!("{p}.conv2"), mid, mid, 3, false);
            bn(&mut w, &mut r, &format!("{p}.bn2"), mid);
            w.conv(&mut r, &format!("{p}.conv3"), out, mid, 1, false);
            bn(&mut w, &mut r, &format!("{p}.bn3"), out);
            if bi == 0 {
                w.conv(&mut r, &format!("{p}.downsample.0"), out, cin, 1, false);
                bn(&mut w, &mut r, &format!("{p}.downsample.1"), out);
            }
            cin = out;
        }
        level_out.push(out);
    }
    let c = 8usize;
    for (i, &lo) in level_out[1..].iter().enumerate() {
        w.conv(&mut r, &format!("fpn.output{}.0", i + 1), c, lo, 1, false);
        bn(&mut w, &mut r, &format!("fpn.output{}.1", i + 1), c);
    }
    for m in ["fpn.merge1", "fpn.merge2"] {
        w.conv(&mut r, &format!("{m}.0"), c, c, 3, false);
        bn(&mut w, &mut r, &format!("{m}.1"), c);
    }
    for s in ["ssh1", "ssh2", "ssh3"] {
        for (name, o, i) in [("conv3X3", c / 2, c), ("conv5X5_1", c / 4, c), ("conv5X5_2", c / 4, c / 4), ("conv7X7_2", c / 4, c / 4), ("conv7x7_3", c / 4, c / 4)] {
            w.conv(&mut r, &format!("{s}.{name}.0"), o, i, 3, false);
            bn(&mut w, &mut r, &format!("{s}.{name}.1"), o);
        }
    }
    for i in 0..3 {
        w.conv(&mut r, &format!("ClassHead.{i}.conv1x1"), 4, c, 1, true);
        w.conv(&mut r, &format!("BboxHead.{i}.conv1x1"), 8, c, 1, true);
        w.conv(&mut r, &format!("LandmarkHead.{i}.conv1x1"), 20, c, 1, true);
    }
    let weights = w.build();
    use clearframe_inference::arch::retinaface as rf;
    let cpu = rf::RetinaFace::load(&weights).unwrap();
    let gnet = clearframe_inference::gpu::nets::GRetinaFace::new(&g, &cpu).unwrap();
    let img = image(&mut r, 160, 96).to_tensor(&Device::Cpu, clearframe_inference::DType::F32).unwrap();
    let x = rf::preprocess(&img).unwrap();
    let (a_loc, a_conf, a_lm) = cpu.forward(&x).unwrap();
    let o = gnet.forward(&g, &clearframe_inference::gpu::nets::Planar::from_tensor(&x).unwrap()).unwrap();
    let (b_loc, b_conf, b_lm) = rf::heads_to_outputs(&o[0..3], &o[3..6], &o[6..9]).unwrap();
    for (name, a, b) in [("loc", a_loc, b_loc), ("conf", a_conf, b_conf), ("landmarks", a_lm, b_lm)] {
        assert_eq!(a.dims(), b.dims(), "{name} shape");
        let (a, b) = (a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let spread = a.iter().map(|v| (v - a[0]).abs()).fold(0f32, f32::max);
        eprintln!("RetinaFace {name}: {} values, max diff {max:.2e}, |output| up to {peak:.2}, spread {spread:.2}", a.len());
        assert!(spread > 1e-3, "{name} output is flat; the test would prove nothing");
        assert!(max <= 1e-3 * peak.max(1.0), "GPU RetinaFace {name} differs from CPU by {max}");
    }
}
