//! RIFE IFNet (v4.26 "HDv3") frame interpolation.

use candle_core::{Result, Tensor};

use crate::ops::{leaky_relu, pad_replicate, pixel_shuffle, resize_bilinear_factor, warp, Act, Conv2d, ConvTranspose2d};
use crate::weights::Weights;

struct Head {
    cnn0: Conv2d,
    cnn1: Conv2d,
    cnn2: Conv2d,
    cnn3: ConvTranspose2d,
}

impl Head {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            cnn0: Conv2d::load(w, "encode.cnn0", 2, 1)?,
            cnn1: Conv2d::load(w, "encode.cnn1", 1, 1)?,
            cnn2: Conv2d::load(w, "encode.cnn2", 1, 1)?,
            cnn3: ConvTranspose2d::load(w, "encode.cnn3", 2, 1)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.cnn0.forward_act(x, Act::LeakyRelu(0.2))?;
        let x = self.cnn1.forward_act(&x, Act::LeakyRelu(0.2))?;
        let x = self.cnn2.forward_act(&x, Act::LeakyRelu(0.2))?;
        self.cnn3.forward(&x)
    }
}

struct IfBlock {
    conv0: [Conv2d; 2],
    blocks: Vec<(Conv2d, Tensor)>,
    last: ConvTranspose2d,
}

impl IfBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let n = w.count_indexed(&format!("{p}.convblock."));
        let mut blocks = Vec::new();
        for i in 0..n {
            blocks.push((Conv2d::load(w, &format!("{p}.convblock.{i}.conv"), 1, 1)?, w.get(&format!("{p}.convblock.{i}.beta"))?));
        }
        Ok(Self {
            conv0: [Conv2d::load(w, &format!("{p}.conv0.0.0"), 2, 1)?, Conv2d::load(w, &format!("{p}.conv0.1.0"), 2, 1)?],
            blocks,
            last: ConvTranspose2d::load(w, &format!("{p}.lastconv.0"), 2, 1)?,
        })
    }

    fn forward(&self, x: &Tensor, flow: Option<&Tensor>, scale: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let mut x = resize_bilinear_factor(x, 1.0 / scale)?;
        if let Some(f) = flow {
            let f = (resize_bilinear_factor(f, 1.0 / scale)? * (1.0 / scale))?;
            x = Tensor::cat(&[&x, &f], 1)?;
        }
        let mut feat = self.conv0[0].forward_act(&x, Act::LeakyRelu(0.2))?;
        feat = self.conv0[1].forward_act(&feat, Act::LeakyRelu(0.2))?;
        for (conv, beta) in &self.blocks {
            feat = leaky_relu(&(conv.forward(&feat)?.broadcast_mul(beta)? + &feat)?, 0.2)?;
        }
        let tmp = pixel_shuffle(&self.last.forward(&feat)?, 2)?;
        let tmp = resize_bilinear_factor(&tmp, scale)?;
        let flow = (tmp.narrow(1, 0, 4)? * scale)?;
        let mask = tmp.narrow(1, 4, 1)?;
        let feat = tmp.narrow(1, 5, tmp.dim(1)? - 5)?;
        Ok((flow, mask, feat))
    }
}

pub struct Rife {
    encode: Head,
    blocks: Vec<IfBlock>,
}

impl Rife {
    pub fn load(w: &Weights) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..8 {
            let p = format!("block{i}");
            if !w.contains(&format!("{p}.conv0.0.0.weight")) {
                break;
            }
            blocks.push(IfBlock::load(w, &p)?);
        }
        if blocks.len() != 5 {
            candle_core::bail!("unsupported RIFE checkpoint layout ({} blocks)", blocks.len());
        }
        Ok(Self { encode: Head::load(w)?, blocks })
    }

    /// Interpolate between `img0` and `img1` (1,3,H,W in 0..1) at `timestep` in (0,1).
    /// `scale` is RIFE's resolution scale (1.0 default; 0.5 recommended for 4K).
    pub fn interpolate(&self, img0: &Tensor, img1: &Tensor, timestep: f32, scale: f64) -> Result<Tensor> {
        let (_, _, h, w) = img0.dims4()?;
        let m = ((64.0 / scale).max(64.0)) as usize;
        let ph = (m - h % m) % m;
        let pw = (m - w % m) % m;
        // Upstream inference pads with zeros (F.pad); we replicate edges to avoid dark borders being warped in.
        let i0 = pad_replicate(img0, pw, ph)?;
        let i1 = pad_replicate(img1, pw, ph)?;
        let out = self.forward(&i0, &i1, timestep, scale)?;
        out.narrow(2, 0, h)?.narrow(3, 0, w)
    }

    /// Raw network forward on padded inputs.
    pub fn forward(&self, img0: &Tensor, img1: &Tensor, timestep: f32, scale: f64) -> Result<Tensor> {
        let (_, _, h, w) = img0.dims4()?;
        let ts = Tensor::full(timestep, (1, 1, h, w), img0.device())?.to_dtype(img0.dtype())?;
        let f0 = self.encode.forward(img0)?;
        let f1 = self.encode.forward(img1)?;
        let scales = [16.0 / scale, 8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale];
        let mut flow: Option<Tensor> = None;
        let mut mask: Option<Tensor> = None;
        let mut feat: Option<Tensor> = None;
        let mut wi0 = img0.clone();
        let mut wi1 = img1.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            match flow.take() {
                None => {
                    let inp = Tensor::cat(&[img0, img1, &f0, &f1, &ts], 1)?;
                    let (fl, m, ft) = block.forward(&inp, None, scales[i])?;
                    flow = Some(fl);
                    mask = Some(m);
                    feat = Some(ft);
                }
                Some(fl) => {
                    let wf0 = warp(&f0, &fl.narrow(1, 0, 2)?)?;
                    let wf1 = warp(&f1, &fl.narrow(1, 2, 2)?)?;
                    let inp = Tensor::cat(&[&wi0, &wi1, &wf0, &wf1, &ts, mask.as_ref().unwrap(), feat.as_ref().unwrap()], 1)?;
                    let (fd, m, ft) = block.forward(&inp, Some(&fl), scales[i])?;
                    flow = Some((fl + fd)?);
                    mask = Some(m);
                    feat = Some(ft);
                }
            }
            let fl = flow.as_ref().unwrap();
            wi0 = warp(img0, &fl.narrow(1, 0, 2)?)?;
            wi1 = warp(img1, &fl.narrow(1, 2, 2)?)?;
        }
        let m = candle_nn::ops::sigmoid(mask.as_ref().unwrap())?;
        let out = (wi0.broadcast_mul(&m)? + wi1.broadcast_mul(&(1.0 - &m)?)?)?;
        out.clamp(0f32, 1f32)
    }
}
