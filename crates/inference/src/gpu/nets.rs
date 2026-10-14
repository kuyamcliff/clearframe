//! GPU implementations of the image-to-image networks.
//!
//! Each network is built from the already-loaded CPU model (so weights go through
//! the same reviewed loader) and mirrors its `forward` exactly: same padding,
//! same layer order, same output size. Small pre/post-processing steps (edge
//! padding, the FFDNet/DRUNet noise map) happen on the CPU around the upload.

use std::sync::Arc;

use candle_core::{DType, Tensor};

use super::{GAct, GConv, GTensor, Gpu, Recorder};
use crate::arch::{drunet::DruNet, ffdnet::FfdNet, nafnet::NafNet, rrdb::RrdbNet, srvgg::SrVggCompact};
use crate::ops::{Conv2d, ConvTranspose2d};

type R<T> = Result<T, String>;

fn vec(t: &Tensor) -> R<Vec<f32>> {
    t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())
}

fn gconv(gpu: &Gpu, c: &Conv2d) -> R<GConv> {
    let (cout, cin_g, k, k2) = c.weight.dims4().map_err(|e| e.to_string())?;
    if k != k2 {
        return Err("non-square kernels are not supported on the GPU".into());
    }
    let w = vec(&c.weight)?;
    let b = match &c.bias {
        Some(b) => Some(vec(b)?),
        None => None,
    };
    Ok(GConv::new(gpu, &w, b.as_deref(), cout as u32, cin_g as u32, k as u32, c.stride as u32, c.padding as u32, c.dilation as u32, c.groups as u32))
}

/// Planar (C, H, W) image data.
pub struct Planar {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Planar {
    pub fn from_tensor(t: &Tensor) -> R<Self> {
        let (_, c, h, w) = t.dims4().map_err(|e| e.to_string())?;
        Ok(Self { c, h, w, data: vec(t)? })
    }

    pub fn to_tensor(&self) -> R<Tensor> {
        Tensor::from_slice(&self.data, (1, self.c, self.h, self.w), &candle_core::Device::Cpu).map_err(|e| e.to_string())
    }

    /// Replicate the last row/column to add `right` columns and `bottom` rows.
    pub(crate) fn pad_replicate(&self, right: usize, bottom: usize) -> Self {
        let (h, w) = (self.h + bottom, self.w + right);
        let mut data = vec![0f32; self.c * h * w];
        for c in 0..self.c {
            for y in 0..h {
                let sy = y.min(self.h - 1);
                for x in 0..w {
                    data[(c * h + y) * w + x] = self.data[(c * self.h + sy) * self.w + x.min(self.w - 1)];
                }
            }
        }
        Self { c: self.c, h, w, data }
    }

    fn pad_zero(&self, right: usize, bottom: usize) -> Self {
        let (h, w) = (self.h + bottom, self.w + right);
        let mut data = vec![0f32; self.c * h * w];
        for c in 0..self.c {
            for y in 0..self.h {
                let s = (c * self.h + y) * self.w;
                let d = (c * h + y) * w;
                data[d..d + self.w].copy_from_slice(&self.data[s..s + self.w]);
            }
        }
        Self { c: self.c, h, w, data }
    }

    fn crop(&self, h: usize, w: usize) -> Self {
        if h == self.h && w == self.w {
            return Self { c: self.c, h, w, data: self.data.clone() };
        }
        let mut data = Vec::with_capacity(self.c * h * w);
        for c in 0..self.c {
            for y in 0..h {
                let s = (c * self.h + y) * self.w;
                data.extend_from_slice(&self.data[s..s + w]);
            }
        }
        Self { c: self.c, h, w, data }
    }

    /// Append a constant channel (the noise-level map).
    fn with_constant_channel(mut self, v: f32) -> Self {
        self.data.extend(std::iter::repeat_n(v, self.h * self.w));
        self.c += 1;
        self
    }

    fn pixel_unshuffle2(&self) -> Self {
        let (h, w) = (self.h / 2, self.w / 2);
        let mut data = vec![0f32; self.c * 4 * h * w];
        for c in 0..self.c {
            for i in 0..2 {
                for j in 0..2 {
                    let oc = c * 4 + i * 2 + j;
                    for y in 0..h {
                        for x in 0..w {
                            data[(oc * h + y) * w + x] = self.data[(c * self.h + y * 2 + i) * self.w + x * 2 + j];
                        }
                    }
                }
            }
        }
        Self { c: self.c * 4, h, w, data }
    }
}

fn upload(rec: &mut Recorder, p: &Planar) -> R<GTensor> {
    rec.upload(p.c as u32, p.h as u32, p.w as u32, &p.data)
}

fn read(rec: Recorder, t: &GTensor) -> R<Planar> {
    let (c, h, w) = (t.c as usize, t.h as usize, t.w as usize);
    let data = rec.finish_read(t)?;
    Ok(Planar { c, h, w, data: data[..c * h * w].to_vec() })
}

// ---------------------------------------------------------------- SRVGG

pub struct GSrVgg {
    convs: Vec<GConv>,
    slopes: Vec<wgpu::Buffer>,
    scale: u32,
}

impl GSrVgg {
    pub fn new(gpu: &Gpu, n: &SrVggCompact) -> R<Self> {
        Ok(Self { convs: n.convs.iter().map(|c| gconv(gpu, c)).collect::<R<_>>()?, slopes: n.acts.iter().map(|a| gpu.constant(a)).collect(), scale: n.scale as u32 })
    }

    fn forward(&self, gpu: &Arc<Gpu>, x: &Planar) -> R<Planar> {
        let mut rec = Recorder::new(gpu);
        let input = upload(&mut rec, x)?;
        let mut out: Option<GTensor> = None;
        for (i, c) in self.convs.iter().enumerate() {
            let act = self.slopes.get(i).map(GAct::Prelu).unwrap_or(GAct::None);
            let src = out.as_ref().unwrap_or(&input);
            let next = rec.conv(c, src.all(), act)?;
            out = Some(next);
        }
        let ps = rec.pixel_shuffle(out.as_ref().unwrap().all(), self.scale)?;
        let up = rec.upsample_nearest(input.all(), self.scale)?;
        rec.axpy_into(up.all(), 1.0, ps.all(), 1.0);
        read(rec, &ps)
    }
}

// ---------------------------------------------------------------- RRDB

pub struct GRrdb {
    first: GConv,
    body: Vec<[[GConv; 5]; 3]>,
    conv_body: GConv,
    up1: GConv,
    up2: GConv,
    hr: GConv,
    last: GConv,
    scale: u32,
}

impl GRrdb {
    pub fn new(gpu: &Gpu, n: &RrdbNet) -> R<Self> {
        let rdb = |r: &crate::arch::rrdb::Rdb| -> R<[GConv; 5]> {
            Ok([gconv(gpu, &r.convs[0])?, gconv(gpu, &r.convs[1])?, gconv(gpu, &r.convs[2])?, gconv(gpu, &r.convs[3])?, gconv(gpu, &r.convs[4])?])
        };
        Ok(Self {
            first: gconv(gpu, &n.conv_first)?,
            body: n.body.iter().map(|b| Ok([rdb(&b[0])?, rdb(&b[1])?, rdb(&b[2])?])).collect::<R<_>>()?,
            conv_body: gconv(gpu, &n.conv_body)?,
            up1: gconv(gpu, &n.conv_up1)?,
            up2: gconv(gpu, &n.conv_up2)?,
            hr: gconv(gpu, &n.conv_hr)?,
            last: gconv(gpu, &n.conv_last)?,
            scale: n.scale as u32,
        })
    }

    fn forward(&self, gpu: &Arc<Gpu>, x: &Planar) -> R<Planar> {
        const L: GAct<'static> = GAct::Leaky(0.2);
        let mut rec = Recorder::new(gpu);
        let input = upload(&mut rec, x)?;
        let feat_in = match self.scale {
            2 => Some(rec.pixel_unshuffle(input.all(), 2)?),
            1 => Some(rec.pixel_unshuffle(input.all(), 4)?),
            _ => None,
        };
        let feat = rec.conv(&self.first, feat_in.as_ref().unwrap_or(&input).all(), GAct::None)?;
        drop(feat_in);
        let nf = self.first.cout;
        let gc = self.body.first().map(|b| b[0][0].cout).unwrap_or(32);
        let (h, w) = (feat.h, feat.w);
        let total = nf + 4 * gc;
        // `b` is the running trunk value; `cat` holds [rdb input | x1 | x2 | x3 | x4].
        let b = GTensor::new(gpu, nf, h, w)?;
        rec.axpy_into(feat.all(), 1.0, b.all(), 0.0);
        let mut cur = GTensor::new(gpu, total, h, w)?;
        let mut nxt = GTensor::new(gpu, total, h, w)?;
        let growth = GTensor::new(gpu, gc, h, w)?;
        rec.axpy_into(feat.all(), 1.0, cur.ch(0, nf), 0.0);
        for rrdb in &self.body {
            for convs in rrdb {
                for (i, c) in convs.iter().take(4).enumerate() {
                    let i = i as u32;
                    rec.conv_into(c, cur.ch(0, nf + i * gc), L, growth.all(), None, 1.0);
                    rec.axpy_into(growth.all(), 1.0, cur.ch(nf + i * gc, gc), 0.0);
                }
                // x5 * 0.2 + x → next block's input.
                rec.conv_into(&convs[4], cur.all(), GAct::None, nxt.ch(0, nf), Some((cur.ch(0, nf), 1.0)), 0.2);
                std::mem::swap(&mut cur, &mut nxt);
            }
            // RRDB residual: b = b + 0.2 * out, then out becomes the next input.
            rec.axpy_into(cur.ch(0, nf), 0.2, b.all(), 1.0);
            rec.axpy_into(b.all(), 1.0, cur.ch(0, nf), 0.0);
        }
        drop((nxt, growth));
        let trunk = rec.conv_res(&self.conv_body, b.all(), GAct::None, feat.all(), 1.0, 1.0)?;
        drop((b, cur, feat));
        let u = rec.upsample_nearest(trunk.all(), 2)?;
        drop(trunk);
        let f = rec.conv(&self.up1, u.all(), L)?;
        drop(u);
        let u = rec.upsample_nearest(f.all(), 2)?;
        drop(f);
        let f = rec.conv(&self.up2, u.all(), L)?;
        drop(u);
        let f2 = rec.conv(&self.hr, f.all(), L)?;
        drop(f);
        let out = rec.conv(&self.last, f2.all(), GAct::None)?;
        drop(f2);
        read(rec, &out)
    }
}

// ---------------------------------------------------------------- FFDNet

pub struct GFfd {
    convs: Vec<GConv>,
}

impl GFfd {
    pub fn new(gpu: &Gpu, n: &FfdNet) -> R<Self> {
        Ok(Self { convs: n.convs.iter().map(|c| gconv(gpu, c)).collect::<R<_>>()? })
    }

    fn forward(&self, gpu: &Arc<Gpu>, x: &Planar, sigma: f32) -> R<Planar> {
        let xin = x.pad_replicate(x.w % 2, x.h % 2).pixel_unshuffle2().with_constant_channel(sigma);
        let mut rec = Recorder::new(gpu);
        let mut y = upload(&mut rec, &xin)?;
        let last = self.convs.len() - 1;
        for (i, c) in self.convs.iter().enumerate() {
            y = rec.conv(c, y.all(), if i != last { GAct::Relu } else { GAct::None })?;
        }
        let out = rec.pixel_shuffle(y.all(), 2)?;
        drop(y);
        Ok(read(rec, &out)?.crop(x.h, x.w))
    }
}

// ---------------------------------------------------------------- DRUNet

pub struct GDru {
    head: GConv,
    downs: Vec<(Vec<[GConv; 2]>, GConv)>,
    body: Vec<[GConv; 2]>,
    ups: Vec<(wgpu::Buffer, u32, Vec<[GConv; 2]>)>,
    tail: GConv,
}

impl GDru {
    pub fn new(gpu: &Gpu, n: &DruNet) -> R<Self> {
        let blocks = |v: &Vec<crate::arch::drunet::ResBlock>| -> R<Vec<[GConv; 2]>> { v.iter().map(|b| Ok([gconv(gpu, &b.c1)?, gconv(gpu, &b.c2)?])).collect() };
        let convt = |t: &ConvTranspose2d| -> R<(wgpu::Buffer, u32)> {
            let (_, cout, k, _) = t.weight.dims4().map_err(|e| e.to_string())?;
            if k != 2 || t.stride != 2 || t.padding != 0 || t.bias.is_some() {
                return Err("unsupported transposed convolution".into());
            }
            Ok((gpu.constant(&vec(&t.weight)?), cout as u32))
        };
        Ok(Self {
            head: gconv(gpu, &n.head)?,
            downs: n.downs.iter().map(|(b, d)| Ok((blocks(b)?, gconv(gpu, d)?))).collect::<R<_>>()?,
            body: blocks(&n.body)?,
            ups: n
                .ups
                .iter()
                .map(|(t, b)| {
                    let (w, c) = convt(t)?;
                    Ok((w, c, blocks(b)?))
                })
                .collect::<R<_>>()?,
            tail: gconv(gpu, &n.tail)?,
        })
    }

    fn res_block(rec: &mut Recorder, b: &[GConv; 2], x: GTensor) -> R<GTensor> {
        let r = rec.conv(&b[0], x.all(), GAct::Relu)?;
        rec.conv_res(&b[1], r.all(), GAct::None, x.all(), 1.0, 1.0)
    }

    fn forward(&self, gpu: &Arc<Gpu>, x: &Planar, sigma: f32) -> R<Planar> {
        let (ph, pw) = ((8 - x.h % 8) % 8, (8 - x.w % 8) % 8);
        let xin = x.pad_replicate(pw, ph).with_constant_channel(sigma);
        let mut rec = Recorder::new(gpu);
        let x0 = upload(&mut rec, &xin)?;
        let x1 = rec.conv(&self.head, x0.all(), GAct::None)?;
        drop(x0);
        // skips[0] = head output, skips[1..=3] = output of each down-sampling step.
        let copy = |rec: &mut Recorder, t: &GTensor| -> R<GTensor> {
            let keep = GTensor::new(gpu, t.c, t.h, t.w)?;
            rec.axpy_into(t.all(), 1.0, keep.all(), 0.0);
            Ok(keep)
        };
        let mut skips = vec![copy(&mut rec, &x1)?];
        let mut y = x1;
        for (blocks, down) in &self.downs {
            for b in blocks {
                y = Self::res_block(&mut rec, b, y)?;
            }
            y = rec.conv(down, y.all(), GAct::None)?;
            skips.push(copy(&mut rec, &y)?);
        }
        for b in &self.body {
            y = Self::res_block(&mut rec, b, y)?;
        }
        for (i, (wt, cout, blocks)) in self.ups.iter().enumerate() {
            let s = rec.add(y.all(), skips[3 - i].all())?;
            y = rec.conv_transpose2x2(s.all(), wt, *cout)?;
            drop(s);
            for b in blocks {
                y = Self::res_block(&mut rec, b, y)?;
            }
        }
        let s = rec.add(y.all(), skips[0].all())?;
        drop((y, skips));
        let out = rec.conv(&self.tail, s.all(), GAct::None)?;
        drop(s);
        Ok(read(rec, &out)?.crop(x.h, x.w))
    }
}

// ---------------------------------------------------------------- NAFNet

struct GNafBlock {
    n1: (wgpu::Buffer, wgpu::Buffer),
    n2: (wgpu::Buffer, wgpu::Buffer),
    conv1: GConv,
    conv2: GConv,
    conv3: GConv,
    sca: GConv,
    conv4: GConv,
    conv5: GConv,
    beta: wgpu::Buffer,
    gamma: wgpu::Buffer,
}

pub struct GNaf {
    intro: GConv,
    ending: GConv,
    encoders: Vec<Vec<GNafBlock>>,
    downs: Vec<GConv>,
    middle: Vec<GNafBlock>,
    ups: Vec<GConv>,
    decoders: Vec<Vec<GNafBlock>>,
}

impl GNaf {
    pub fn new(gpu: &Gpu, n: &NafNet) -> R<Self> {
        let block = |b: &crate::arch::nafnet::NafBlock| -> R<GNafBlock> {
            Ok(GNafBlock {
                n1: (gpu.constant(&vec(&b.norm1.0)?), gpu.constant(&vec(&b.norm1.1)?)),
                n2: (gpu.constant(&vec(&b.norm2.0)?), gpu.constant(&vec(&b.norm2.1)?)),
                conv1: gconv(gpu, &b.conv1)?,
                conv2: gconv(gpu, &b.conv2)?,
                conv3: gconv(gpu, &b.conv3)?,
                sca: gconv(gpu, &b.sca)?,
                conv4: gconv(gpu, &b.conv4)?,
                conv5: gconv(gpu, &b.conv5)?,
                beta: gpu.constant(&vec(&b.beta)?),
                gamma: gpu.constant(&vec(&b.gamma)?),
            })
        };
        let blocks = |v: &Vec<crate::arch::nafnet::NafBlock>| -> R<Vec<GNafBlock>> { v.iter().map(block).collect() };
        Ok(Self {
            intro: gconv(gpu, &n.intro)?,
            ending: gconv(gpu, &n.ending)?,
            encoders: n.encoders.iter().map(blocks).collect::<R<_>>()?,
            downs: n.downs.iter().map(|c| gconv(gpu, c)).collect::<R<_>>()?,
            middle: blocks(&n.middle)?,
            ups: n.ups.iter().map(|c| gconv(gpu, c)).collect::<R<_>>()?,
            decoders: n.decoders.iter().map(blocks).collect::<R<_>>()?,
        })
    }

    fn block(rec: &mut Recorder, b: &GNafBlock, inp: GTensor) -> R<GTensor> {
        let x = rec.layer_norm(inp.all(), &b.n1.0, &b.n1.1, 1e-6)?;
        let x = rec.conv(&b.conv1, x.all(), GAct::None)?;
        let x = rec.conv(&b.conv2, x.all(), GAct::None)?;
        let x = rec.gate(x.all())?;
        let pooled = rec.avg_pool(x.all())?;
        let att = rec.conv(&b.sca, pooled.all(), GAct::None)?;
        drop(pooled);
        let attb = rec.buffer_of(&att).clone();
        let x = rec.scale_channels(x.all(), &attb, None)?;
        drop(att);
        let x = rec.conv(&b.conv3, x.all(), GAct::None)?;
        let y = rec.scale_channels(x.all(), &b.beta, Some(inp.all()))?;
        drop((x, inp));
        let x = rec.layer_norm(y.all(), &b.n2.0, &b.n2.1, 1e-6)?;
        let x = rec.conv(&b.conv4, x.all(), GAct::None)?;
        let x = rec.gate(x.all())?;
        let x = rec.conv(&b.conv5, x.all(), GAct::None)?;
        rec.scale_channels(x.all(), &b.gamma, Some(y.all()))
    }

    fn forward(&self, gpu: &Arc<Gpu>, x: &Planar) -> R<Planar> {
        let m = 1usize << self.encoders.len();
        let xp = x.pad_zero((m - x.w % m) % m, (m - x.h % m) % m);
        let mut rec = Recorder::new(gpu);
        let inp = upload(&mut rec, &xp)?;
        let mut y = rec.conv(&self.intro, inp.all(), GAct::None)?;
        let mut encs = Vec::new();
        for (enc, down) in self.encoders.iter().zip(&self.downs) {
            for b in enc {
                y = Self::block(&mut rec, b, y)?;
            }
            let keep = GTensor::new(gpu, y.c, y.h, y.w)?;
            rec.axpy_into(y.all(), 1.0, keep.all(), 0.0);
            encs.push(keep);
            y = rec.conv(down, y.all(), GAct::None)?;
        }
        for b in &self.middle {
            y = Self::block(&mut rec, b, y)?;
        }
        for ((dec, up), skip) in self.decoders.iter().zip(&self.ups).zip(encs.iter().rev()) {
            let u = rec.conv(up, y.all(), GAct::None)?;
            let s = rec.pixel_shuffle(u.all(), 2)?;
            drop(u);
            y = rec.add(s.all(), skip.all())?;
            for b in dec {
                y = Self::block(&mut rec, b, y)?;
            }
        }
        drop(encs);
        let out = rec.conv_res(&self.ending, y.all(), GAct::None, inp.all(), 1.0, 1.0)?;
        drop((y, inp));
        Ok(read(rec, &out)?.crop(x.h, x.w))
    }
}

// ---------------------------------------------------------------- dispatch

/// A GPU network for one of the image-to-image architectures.
pub enum GpuNet {
    SrVgg(GSrVgg),
    Rrdb(Box<GRrdb>),
    Ffd(GFfd),
    Dru(Box<GDru>),
    Naf(Box<GNaf>),
}

impl GpuNet {
    /// Run on (1, 3, H, W) input, matching the CPU network's `forward`.
    pub fn forward(&self, gpu: &Arc<Gpu>, x: &Tensor, sigma: f32) -> R<Tensor> {
        let p = Planar::from_tensor(x)?;
        let out = match self {
            GpuNet::SrVgg(n) => n.forward(gpu, &p)?,
            GpuNet::Rrdb(n) => n.forward(gpu, &p)?,
            GpuNet::Ffd(n) => n.forward(gpu, &p, sigma)?,
            GpuNet::Dru(n) => n.forward(gpu, &p, sigma)?,
            GpuNet::Naf(n) => n.forward(gpu, &p)?,
        };
        gpu.trim();
        out.to_tensor()
    }

    /// Peak working memory for one (h, w) input tile, in bytes (largest single tensor).
    pub fn largest_tensor_bytes(&self, h: usize, w: usize) -> u64 {
        let px = (h * w) as u64 * 4;
        match self {
            GpuNet::SrVgg(n) => px * n.convs.iter().map(|c| c.cout as u64).max().unwrap_or(64),
            GpuNet::Rrdb(n) => {
                // After the second up-sampling: nf channels at `scale`× the input size.
                let s = n.scale as u64;
                px * n.first.cout as u64 * s * s
            }
            GpuNet::Ffd(n) => px / 4 * n.convs.iter().map(|c| c.cout as u64).max().unwrap_or(96),
            GpuNet::Dru(_) => px * 64,
            GpuNet::Naf(n) => px * n.intro.cout as u64 * 2,
        }
    }
}

// ---------------------------------------------------------------- RIFE (frame interpolation)

/// A transposed convolution prepared for the GPU.
struct GConvT {
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
}

impl GConvT {
    fn new(gpu: &Gpu, c: &ConvTranspose2d) -> R<Self> {
        let (_cin, cout, k, k2) = c.weight.dims4().map_err(|e| e.to_string())?;
        if k != k2 {
            return Err("non-square transposed kernels are not supported on the GPU".into());
        }
        let bias = match &c.bias {
            Some(b) => vec(b)?,
            None => vec![0.0; cout],
        };
        Ok(Self { weight: gpu.constant(&vec(&c.weight)?), bias: gpu.constant(&bias), cout: cout as u32, k: k as u32, stride: c.stride as u32, pad: c.padding as u32 })
    }
    fn run(&self, r: &mut Recorder, x: super::View<'_>) -> R<GTensor> {
        r.conv_transpose(x, &self.weight, &self.bias, self.cout, self.k, self.stride, self.pad)
    }
}

struct GIfBlock {
    conv0: [GConv; 2],
    blocks: Vec<(GConv, wgpu::Buffer)>,
    last: GConvT,
}

/// RIFE v4.26 IFNet on the GPU; mirrors `arch::rife::Rife::forward`.
pub struct GRife {
    head: [GConv; 3],
    head_up: GConvT,
    blocks: Vec<GIfBlock>,
}

/// Concatenate channel views into a new tensor.
fn cat(r: &mut Recorder, parts: &[super::View<'_>]) -> R<GTensor> {
    let (h, w) = (parts[0].t.h, parts[0].t.w);
    let c: u32 = parts.iter().map(|v| v.len).sum();
    let out = GTensor::new(r.gpu(), c, h, w)?;
    let mut off = 0;
    for v in parts {
        assert!(v.t.h == h && v.t.w == w, "cat size mismatch");
        r.axpy_into(*v, 1.0, out.ch(off, v.len), 0.0);
        off += v.len;
    }
    Ok(out)
}

impl GRife {
    pub fn new(gpu: &Arc<Gpu>, n: &crate::arch::rife::Rife) -> R<Self> {
        let e = &n.encode;
        let blocks = n
            .blocks
            .iter()
            .map(|b| {
                Ok(GIfBlock {
                    conv0: [gconv(gpu, &b.conv0[0])?, gconv(gpu, &b.conv0[1])?],
                    blocks: b.blocks.iter().map(|(c, beta)| Ok((gconv(gpu, c)?, gpu.constant(&vec(beta)?)))).collect::<R<Vec<_>>>()?,
                    last: GConvT::new(gpu, &b.last)?,
                })
            })
            .collect::<R<Vec<_>>>()?;
        Ok(Self { head: [gconv(gpu, &e.cnn0)?, gconv(gpu, &e.cnn1)?, gconv(gpu, &e.cnn2)?], head_up: GConvT::new(gpu, &e.cnn3)?, blocks })
    }

    fn encode(&self, r: &mut Recorder, x: &GTensor) -> R<GTensor> {
        let l = GAct::Leaky(0.2);
        let a = r.conv(&self.head[0], x.all(), l)?;
        let a = r.conv(&self.head[1], a.all(), l)?;
        let a = r.conv(&self.head[2], a.all(), l)?;
        self.head_up.run(r, a.all())
    }

    /// Returns (flow 4ch, mask 1ch, feat) at the input resolution.
    fn block(&self, r: &mut Recorder, b: &GIfBlock, x: &GTensor, flow: Option<&GTensor>, scale: f64) -> R<(GTensor, GTensor, GTensor)> {
        let l = GAct::Leaky(0.2);
        let xs = r.resize_bilinear_factor(x.all(), 1.0 / scale, 1.0)?;
        let inp = match flow {
            Some(f) => {
                let fs = r.resize_bilinear_factor(f.all(), 1.0 / scale, (1.0 / scale) as f32)?;
                cat(r, &[xs.all(), fs.all()])?
            }
            None => xs,
        };
        let mut feat = r.conv(&b.conv0[0], inp.all(), l)?;
        feat = r.conv(&b.conv0[1], feat.all(), l)?;
        for (conv, beta) in &b.blocks {
            let t = r.conv(conv, feat.all(), GAct::None)?;
            feat = r.scale_add_leaky(t.all(), beta, feat.all(), 0.2)?;
        }
        let up = b.last.run(r, feat.all())?;
        let tmp = r.pixel_shuffle(up.all(), 2)?;
        let flow = r.resize_bilinear_factor(tmp.ch(0, 4), scale, scale as f32)?;
        let mask = r.resize_bilinear_factor(tmp.ch(4, 1), scale, 1.0)?;
        let feat = r.resize_bilinear_factor(tmp.ch(5, tmp.c - 5), scale, 1.0)?;
        Ok((flow, mask, feat))
    }

    /// Interpolate between padded planar frames (sizes already a multiple of the RIFE alignment).
    pub fn forward(&self, gpu: &Arc<Gpu>, a: &Planar, b: &Planar, timestep: f32, scale: f64) -> R<Planar> {
        let mut r = Recorder::new(gpu);
        let (h, w) = (a.h as u32, a.w as u32);
        let img0 = r.upload(3, h, w, &a.data)?;
        let img1 = r.upload(3, h, w, &b.data)?;
        let ts = r.upload(1, h, w, &vec![timestep; a.h * a.w])?;
        let f0 = self.encode(&mut r, &img0)?;
        let f1 = self.encode(&mut r, &img1)?;
        let scales = [16.0 / scale, 8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale];
        let (mut flow, mut mask, mut feat): (Option<GTensor>, Option<GTensor>, Option<GTensor>) = (None, None, None);
        let (mut wi0, mut wi1): (Option<GTensor>, Option<GTensor>) = (None, None);
        for (i, blk) in self.blocks.iter().enumerate() {
            match flow.take() {
                None => {
                    let inp = cat(&mut r, &[img0.all(), img1.all(), f0.all(), f1.all(), ts.all()])?;
                    let (fl, m, ft) = self.block(&mut r, blk, &inp, None, scales[i])?;
                    flow = Some(fl);
                    mask = Some(m);
                    feat = Some(ft);
                }
                Some(fl) => {
                    let wf0 = r.warp(f0.all(), fl.ch(0, 2))?;
                    let wf1 = r.warp(f1.all(), fl.ch(2, 2))?;
                    let (w0, w1, m0, ft0) = (wi0.as_ref().unwrap(), wi1.as_ref().unwrap(), mask.as_ref().unwrap(), feat.as_ref().unwrap());
                    let inp = cat(&mut r, &[w0.all(), w1.all(), wf0.all(), wf1.all(), ts.all(), m0.all(), ft0.all()])?;
                    let (fd, m, ft) = self.block(&mut r, blk, &inp, Some(&fl), scales[i])?;
                    flow = Some(r.add(fl.all(), fd.all())?);
                    mask = Some(m);
                    feat = Some(ft);
                }
            }
            let fl = flow.as_ref().unwrap();
            wi0 = Some(r.warp(img0.all(), fl.ch(0, 2))?);
            wi1 = Some(r.warp(img1.all(), fl.ch(2, 2))?);
        }
        let out = r.sigmoid_blend(wi0.as_ref().unwrap().all(), wi1.as_ref().unwrap().all(), mask.as_ref().unwrap().all())?;
        let data = r.finish_read(&out)?;
        Ok(Planar { c: 3, h: a.h, w: a.w, data })
    }
}

// ---------------------------------------------------------------- GFPGAN (face restoration)

/// A StyleGAN2 modulated convolution as `d[o] * conv(x * s[i], W)`: the static weight lives on the GPU and the
/// per-face style only scales input channels (`s`) and output channels (`d`, the demodulation), computed on the CPU.
struct GModConv {
    conv: GConv,
    /// Σ_k W[o,i,k]² as an (out, in) CPU tensor, for the demodulation factor.
    w2: Tensor,
    modulation: crate::ops::Linear,
    demodulate: bool,
    upsample: bool,
}

impl GModConv {
    fn new(gpu: &Gpu, m: &crate::arch::gfpgan::ModConv) -> R<Self> {
        let (cout, cin, k, _) = m.weight.dims4().map_err(|e| e.to_string())?;
        let conv = GConv::new(gpu, &vec(&m.weight)?, None, cout as u32, cin as u32, k as u32, 1, m.padding as u32, 1, 1);
        let w2 = m.weight.to_dtype(DType::F32).and_then(|w| w.sqr()).and_then(|w| w.sum(3)).and_then(|w| w.sum(2)).map_err(|e| e.to_string())?;
        Ok(Self { conv, w2, modulation: m.modulation.clone(), demodulate: m.demodulate, upsample: m.upsample })
    }

    /// Per-face (s, d) for a latent row (1, num_style_feat).
    fn style(&self, gpu: &Gpu, lat: &Tensor) -> R<(wgpu::Buffer, wgpu::Buffer)> {
        let e = |x: candle_core::Error| x.to_string();
        let s = self.modulation.forward(lat).map_err(e)?.flatten_all().map_err(e)?.to_dtype(DType::F32).map_err(e)?;
        let d = if self.demodulate {
            // d[o] = 1 / sqrt(Σ_i s_i² Σ_k W[o,i,k]² + 1e-8)
            let s2 = s.sqr().map_err(e)?.unsqueeze(1).map_err(e)?;
            self.w2.matmul(&s2).map_err(e)?.flatten_all().map_err(e)?.affine(1.0, 1e-8).map_err(e)?.sqrt().map_err(e)?.recip().map_err(e)?
        } else {
            Tensor::ones(self.conv.cout as usize, DType::F32, &candle_core::Device::Cpu).map_err(e)?
        };
        Ok((gpu.constant(&vec(&s)?), gpu.constant(&vec(&d)?)))
    }

    /// The modulated convolution without its output scaling (returned `d` is applied by the caller's `style_out`).
    fn run(&self, r: &mut Recorder, x: &GTensor, s: &wgpu::Buffer) -> R<GTensor> {
        let xs = r.scale_channels(x.all(), s, None)?;
        let xs = if self.upsample { r.resize_bilinear_factor(xs.all(), 2.0, 1.0)? } else { xs };
        r.conv(&self.conv, xs.all(), GAct::None)
    }
}

struct GStyleConv {
    conv: GModConv,
    noise: wgpu::Buffer,
    noise_weight: f32,
    bias: wgpu::Buffer,
}

struct GToRgb {
    conv: GModConv,
    bias: wgpu::Buffer,
    upsample: bool,
}

struct GResBlock {
    conv1: GConv,
    conv2: GConv,
    skip: GConv,
    factor: f64,
}

/// GFPGAN v1.4 (GFPGANv1Clean) on the GPU; mirrors `arch::gfpgan::Gfpgan::forward`.
pub struct GGfpgan {
    first: GConv,
    down: Vec<GResBlock>,
    final_conv: GConv,
    up: Vec<GResBlock>,
    final_linear: crate::ops::Linear,
    cond_scale: Vec<(GConv, GConv)>,
    cond_shift: Vec<(GConv, GConv)>,
    constant: Planar,
    style_conv1: GStyleConv,
    to_rgb1: GToRgb,
    style_convs: Vec<GStyleConv>,
    to_rgbs: Vec<GToRgb>,
    num_style_feat: usize,
}

impl GGfpgan {
    pub fn new(gpu: &Arc<Gpu>, n: &crate::arch::gfpgan::Gfpgan) -> R<Self> {
        let res = |b: &crate::arch::gfpgan::ResBlock| -> R<GResBlock> {
            Ok(GResBlock { conv1: gconv(gpu, &b.conv1)?, conv2: gconv(gpu, &b.conv2)?, skip: gconv(gpu, &b.skip)?, factor: b.factor })
        };
        let sc = |s: &crate::arch::gfpgan::StyleConv| -> R<GStyleConv> {
            Ok(GStyleConv { conv: GModConv::new(gpu, &s.conv)?, noise: gpu.constant(&vec(&s.noise)?), noise_weight: vec(&s.noise_weight)?[0], bias: gpu.constant(&vec(&s.bias)?) })
        };
        let rgb = |t: &crate::arch::gfpgan::ToRgb| -> R<GToRgb> { Ok(GToRgb { conv: GModConv::new(gpu, &t.conv)?, bias: gpu.constant(&vec(&t.bias)?), upsample: t.upsample }) };
        let pair = |v: &Vec<(Conv2d, Conv2d)>| -> R<Vec<(GConv, GConv)>> { v.iter().map(|(a, b)| Ok((gconv(gpu, a)?, gconv(gpu, b)?))).collect() };
        Ok(Self {
            first: gconv(gpu, &n.conv_body_first)?,
            down: n.down.iter().map(res).collect::<R<_>>()?,
            final_conv: gconv(gpu, &n.final_conv)?,
            up: n.up.iter().map(res).collect::<R<_>>()?,
            final_linear: n.final_linear.clone(),
            cond_scale: pair(&n.cond_scale)?,
            cond_shift: pair(&n.cond_shift)?,
            constant: Planar::from_tensor(&n.constant)?,
            style_conv1: sc(&n.style_conv1)?,
            to_rgb1: rgb(&n.to_rgb1)?,
            style_convs: n.style_convs.iter().map(sc).collect::<R<_>>()?,
            to_rgbs: n.to_rgbs.iter().map(rgb).collect::<R<_>>()?,
            num_style_feat: n.num_style_feat,
        })
    }

    fn resblock(r: &mut Recorder, b: &GResBlock, x: &GTensor) -> R<GTensor> {
        let l = GAct::Leaky(0.2);
        let out = r.conv(&b.conv1, x.all(), l)?;
        let out = r.resize_bilinear_factor(out.all(), b.factor, 1.0)?;
        let out = r.conv(&b.conv2, out.all(), l)?;
        let xs = r.resize_bilinear_factor(x.all(), b.factor, 1.0)?;
        let skip = r.conv(&b.skip, xs.all(), GAct::None)?;
        r.add(out.all(), skip.all())
    }

    fn style_conv(&self, gpu: &Gpu, r: &mut Recorder, s: &GStyleConv, x: &GTensor, lat: &Tensor) -> R<GTensor> {
        let (sb, db) = s.conv.style(gpu, lat)?;
        let y = s.conv.run(r, x, &sb)?;
        r.style_out(y.all(), &db, std::f32::consts::SQRT_2, Some((&s.noise, s.noise_weight)), &s.bias, Some(0.2), None)
    }

    fn to_rgb(&self, gpu: &Gpu, r: &mut Recorder, t: &GToRgb, x: &GTensor, lat: &Tensor, skip: Option<&GTensor>) -> R<GTensor> {
        let (sb, db) = t.conv.style(gpu, lat)?;
        let y = t.conv.run(r, x, &sb)?;
        let skip = match skip {
            Some(s) if t.upsample => Some(r.resize_bilinear_factor(s.all(), 2.0, 1.0)?),
            Some(s) => Some(r.resize_bilinear_factor(s.all(), 1.0, 1.0)?),
            None => None,
        };
        r.style_out(y.all(), &db, 1.0, None, &t.bias, None, skip.as_ref().map(|s| s.all()))
    }

    /// Restore an aligned 512×512 face given as planar RGB in 0..1; returns the same layout in 0..1.
    pub fn forward(&self, gpu: &Arc<Gpu>, face: &Planar) -> R<Planar> {
        let e = |x: candle_core::Error| x.to_string();
        // Pass 1: encoder down to the style features.
        let mut r = Recorder::new(gpu);
        let norm: Vec<f32> = face.data.iter().map(|v| (v - 0.5) / 0.5).collect();
        let x = r.upload(3, face.h as u32, face.w as u32, &norm)?;
        let first = r.conv(&self.first, x.all(), GAct::Leaky(0.2))?;
        // Every encoder output stays alive on the GPU: they are the skip connections of pass 2.
        let mut downs: Vec<GTensor> = Vec::new();
        for d in &self.down {
            let next = Self::resblock(&mut r, d, downs.last().unwrap_or(&first))?;
            downs.push(next);
        }
        let feat = r.conv(&self.final_conv, downs.last().unwrap().all(), GAct::Leaky(0.2))?;
        let flat = r.finish_read(&feat)?;
        let style = Tensor::from_vec(flat, (1, feat.c as usize * feat.h as usize * feat.w as usize), &candle_core::Device::Cpu).map_err(e)?;
        let style = self.final_linear.forward(&style).map_err(e)?;
        let n_latent = style.dim(1).map_err(e)? / self.num_style_feat;
        let latent = style.reshape((n_latent, self.num_style_feat)).map_err(e)?;
        let lat = |i: usize| latent.narrow(0, i, 1).map_err(e);
        // Pass 2: up path (SFT conditions) and the StyleGAN decoder.
        let mut r = Recorder::new(gpu);
        let mut feat = feat;
        let mut conditions = Vec::new();
        for (i, u) in self.up.iter().enumerate() {
            let sum = r.add(feat.all(), downs[downs.len() - 1 - i].all())?;
            feat = Self::resblock(&mut r, u, &sum)?;
            let (a, b) = &self.cond_scale[i];
            let t = r.conv(a, feat.all(), GAct::Leaky(0.2))?;
            conditions.push(r.conv(b, t.all(), GAct::None)?);
            let (a, b) = &self.cond_shift[i];
            let t = r.conv(a, feat.all(), GAct::Leaky(0.2))?;
            conditions.push(r.conv(b, t.all(), GAct::None)?);
        }
        let c = &self.constant;
        let constant = r.upload(c.c as u32, c.h as u32, c.w as u32, &c.data)?;
        let mut out = self.style_conv(gpu, &mut r, &self.style_conv1, &constant, &lat(0)?)?;
        let mut skip = self.to_rgb(gpu, &mut r, &self.to_rgb1, &out, &lat(1)?, None)?;
        let mut i = 1;
        for (k, t) in self.to_rgbs.iter().enumerate() {
            out = self.style_conv(gpu, &mut r, &self.style_convs[2 * k], &out, &lat(i)?)?;
            if i < conditions.len() {
                out = r.sft(out.all(), conditions[i - 1].all(), conditions[i].all())?;
            }
            out = self.style_conv(gpu, &mut r, &self.style_convs[2 * k + 1], &out, &lat(i + 1)?)?;
            skip = self.to_rgb(gpu, &mut r, t, &out, &lat(i + 2)?, Some(&skip))?;
            i += 2;
        }
        let data = r.finish_read(&skip)?;
        Ok(Planar { c: 3, h: face.h, w: face.w, data: data.into_iter().map(|v| (v.clamp(-1.0, 1.0) + 1.0) * 0.5).collect() })
    }
}

// ---------------------------------------------------------------- RetinaFace (face detection)

struct GCbr {
    conv: GConv,
    act: Option<f32>,
}

impl GCbr {
    fn new(gpu: &Gpu, c: &crate::arch::retinaface::Cbr) -> R<Self> {
        Ok(Self { conv: gconv(gpu, &c.conv)?, act: c.act.map(|a| a as f32) })
    }
    fn act(&self) -> GAct<'static> {
        match self.act {
            Some(0.0) => GAct::Relu,
            Some(s) => GAct::Leaky(s),
            None => GAct::None,
        }
    }
    fn run(&self, r: &mut Recorder, x: &GTensor) -> R<GTensor> {
        r.conv(&self.conv, x.all(), self.act())
    }
}

/// ResNet bottleneck: 1×1, 3×3, 1×1 and an optional projection for the identity path.
struct GBottleneck {
    c1: GCbr,
    c2: GCbr,
    c3: GCbr,
    down: Option<GCbr>,
}

enum GBackbone {
    ResNet50 { stem: GCbr, layers: Vec<Vec<GBottleneck>> },
    MobileNet { stages: Vec<Vec<GCbr>> },
}

struct GSsh {
    c3: GCbr,
    c5_1: GCbr,
    c5_2: GCbr,
    c7_2: GCbr,
    c7_3: GCbr,
}

/// RetinaFace on the GPU; mirrors `arch::retinaface::RetinaFace::forward` up to the raw head outputs.
pub struct GRetinaFace {
    backbone: GBackbone,
    fpn_out: [GCbr; 3],
    merge1: GCbr,
    merge2: GCbr,
    ssh: [GSsh; 3],
    heads: [[GConv; 3]; 3],
}

impl GRetinaFace {
    pub fn new(gpu: &Arc<Gpu>, n: &crate::arch::retinaface::RetinaFace) -> R<Self> {
        use crate::arch::retinaface::Backbone;
        let c = |x: &crate::arch::retinaface::Cbr| GCbr::new(gpu, x);
        let backbone = match &n.backbone {
            Backbone::ResNet50 { stem, layers } => GBackbone::ResNet50 {
                stem: c(stem)?,
                layers: layers
                    .iter()
                    .map(|l| l.iter().map(|b| Ok(GBottleneck { c1: c(&b.c1)?, c2: c(&b.c2)?, c3: c(&b.c3)?, down: b.down.as_ref().map(c).transpose()? })).collect::<R<Vec<_>>>())
                    .collect::<R<Vec<_>>>()?,
            },
            Backbone::MobileNet { stages } => GBackbone::MobileNet { stages: stages.iter().map(|s| s.iter().map(c).collect::<R<Vec<_>>>()).collect::<R<Vec<_>>>()? },
        };
        let ssh = |s: &crate::arch::retinaface::Ssh| -> R<GSsh> { Ok(GSsh { c3: c(&s.c3)?, c5_1: c(&s.c5_1)?, c5_2: c(&s.c5_2)?, c7_2: c(&s.c7_2)?, c7_3: c(&s.c7_3)? }) };
        let heads = |h: &[Conv2d; 3]| -> R<[GConv; 3]> { Ok([gconv(gpu, &h[0])?, gconv(gpu, &h[1])?, gconv(gpu, &h[2])?]) };
        Ok(Self {
            backbone,
            fpn_out: [c(&n.fpn_out[0])?, c(&n.fpn_out[1])?, c(&n.fpn_out[2])?],
            merge1: c(&n.merge1)?,
            merge2: c(&n.merge2)?,
            ssh: [ssh(&n.ssh[0])?, ssh(&n.ssh[1])?, ssh(&n.ssh[2])?],
            heads: [heads(&n.class_head)?, heads(&n.bbox_head)?, heads(&n.landmark_head)?],
        })
    }

    fn backbone(&self, r: &mut Recorder, x: &GTensor) -> R<Vec<GTensor>> {
        let mut outs = Vec::new();
        match &self.backbone {
            GBackbone::ResNet50 { stem, layers } => {
                let s = stem.run(r, x)?;
                let mut y = r.max_pool3s2(s.all())?;
                for (i, layer) in layers.iter().enumerate() {
                    for b in layer {
                        let t = b.c1.run(r, &y)?;
                        let t = b.c2.run(r, &t)?;
                        let t = b.c3.run(r, &t)?;
                        y = match &b.down {
                            Some(d) => {
                                let id = d.run(r, &y)?;
                                r.add_act(t.all(), Some(id.all()), GAct::Relu)?
                            }
                            None => r.add_act(t.all(), Some(y.all()), GAct::Relu)?,
                        };
                    }
                    if i >= 1 {
                        outs.push(r.add_act(y.all(), None, GAct::None)?);
                    }
                }
            }
            GBackbone::MobileNet { stages } => {
                let mut y = r.add_act(x.all(), None, GAct::None)?;
                for st in stages {
                    for c in st {
                        y = c.run(r, &y)?;
                    }
                    outs.push(r.add_act(y.all(), None, GAct::None)?);
                }
            }
        }
        Ok(outs)
    }

    fn ssh(r: &mut Recorder, s: &GSsh, x: &GTensor) -> R<GTensor> {
        let a = s.c3.run(r, x)?;
        let b1 = s.c5_1.run(r, x)?;
        let b = s.c5_2.run(r, &b1)?;
        let c2 = s.c7_2.run(r, &b1)?;
        let c = s.c7_3.run(r, &c2)?;
        let cat = cat(r, &[a.all(), b.all(), c.all()])?;
        r.add_act(cat.all(), None, GAct::Relu)
    }

    /// Raw head outputs for a preprocessed (BGR, mean-subtracted) planar input: [class×3, bbox×3, landmark×3],
    /// each (1, C, H, W) on the CPU.
    pub fn forward(&self, gpu: &Arc<Gpu>, x: &Planar) -> R<Vec<Tensor>> {
        let mut r = Recorder::new(gpu);
        let input = r.upload(x.c as u32, x.h as u32, x.w as u32, &x.data)?;
        let feats = self.backbone(&mut r, &input)?;
        let o1 = self.fpn_out[0].run(&mut r, &feats[0])?;
        let o2 = self.fpn_out[1].run(&mut r, &feats[1])?;
        let o3 = self.fpn_out[2].run(&mut r, &feats[2])?;
        let up3 = r.resize_nearest(o3.all(), o2.h, o2.w)?;
        let s2 = r.add_act(o2.all(), Some(up3.all()), GAct::None)?;
        let o2 = self.merge2.run(&mut r, &s2)?;
        let up2 = r.resize_nearest(o2.all(), o1.h, o1.w)?;
        let s1 = r.add_act(o1.all(), Some(up2.all()), GAct::None)?;
        let o1 = self.merge1.run(&mut r, &s1)?;
        let fs = [Self::ssh(&mut r, &self.ssh[0], &o1)?, Self::ssh(&mut r, &self.ssh[1], &o2)?, Self::ssh(&mut r, &self.ssh[2], &o3)?];
        let mut outs = Vec::new();
        for heads in &self.heads {
            for (f, h) in fs.iter().zip(heads.iter()) {
                outs.push(r.conv(h, f.all(), GAct::None)?);
            }
        }
        let shapes: Vec<(usize, usize, usize)> = outs.iter().map(|t| (t.c as usize, t.h as usize, t.w as usize)).collect();
        let data = r.finish_read_many(&outs.iter().collect::<Vec<_>>())?;
        data.into_iter().zip(shapes).map(|(d, (c, h, w))| Tensor::from_vec(d, (1, c, h, w), &candle_core::Device::Cpu).map_err(|e| e.to_string())).collect()
    }
}
