//! GFPGAN v1.4 (`GFPGANv1Clean`): U-Net encoder + StyleGAN2 decoder with SFT.
//!
//! Noise injection uses the checkpoint's stored noise buffers (upstream
//! `randomize_noise=False`) so results are deterministic and temporally stable.

use candle_core::{Result, Tensor};

use crate::ops::{conv2d, leaky_relu, resize_bilinear_factor, Conv2d, Linear};
use crate::weights::Weights;

pub(crate) struct ModConv {
    pub(crate) weight: Tensor, // (out, in, k, k)
    pub(crate) modulation: Linear,
    pub(crate) demodulate: bool,
    pub(crate) upsample: bool,
    pub(crate) padding: usize,
}

impl ModConv {
    fn load(w: &Weights, p: &str, demodulate: bool, upsample: bool) -> Result<Self> {
        let weight = w.get(&format!("{p}.weight"))?.squeeze(0)?;
        let k = weight.dim(3)?;
        Ok(Self { weight, modulation: Linear::load(w, &format!("{p}.modulation"))?, demodulate, upsample, padding: k / 2 })
    }
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let in_c = self.weight.dim(1)?;
        let s = self.modulation.forward(style)?.reshape((1, in_c, 1, 1))?;
        let mut wgt = self.weight.broadcast_mul(&s)?;
        if self.demodulate {
            let out_c = wgt.dim(0)?;
            let d = (wgt.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sum_keepdim(3)? + 1e-8)?.sqrt()?.recip()?;
            wgt = wgt.broadcast_mul(&d.reshape((out_c, 1, 1, 1))?)?;
        }
        let x = if self.upsample { resize_bilinear_factor(x, 2.0)? } else { x.clone() };
        conv2d(&x, &wgt, self.padding, 1, 1, 1)
    }
}

pub(crate) struct StyleConv {
    pub(crate) conv: ModConv,
    pub(crate) noise_weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) noise: Tensor,
}

impl StyleConv {
    fn load(w: &Weights, p: &str, upsample: bool, noise: Tensor) -> Result<Self> {
        Ok(Self {
            conv: ModConv::load(w, &format!("{p}.modulated_conv"), true, upsample)?,
            noise_weight: w.get(&format!("{p}.weight"))?,
            bias: w.get(&format!("{p}.bias"))?,
            noise,
        })
    }
    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let out = (self.conv.forward(x, style)? * std::f64::consts::SQRT_2)?;
        let out = out.broadcast_add(&self.noise.broadcast_mul(&self.noise_weight)?)?;
        let out = out.broadcast_add(&self.bias)?;
        leaky_relu(&out, 0.2)
    }
}

pub(crate) struct ToRgb {
    pub(crate) conv: ModConv,
    pub(crate) bias: Tensor,
    pub(crate) upsample: bool,
}

impl ToRgb {
    fn load(w: &Weights, p: &str, upsample: bool) -> Result<Self> {
        Ok(Self { conv: ModConv::load(w, &format!("{p}.modulated_conv"), false, false)?, bias: w.get(&format!("{p}.bias"))?, upsample })
    }
    fn forward(&self, x: &Tensor, style: &Tensor, skip: Option<&Tensor>) -> Result<Tensor> {
        let out = self.conv.forward(x, style)?.broadcast_add(&self.bias)?;
        match skip {
            Some(s) => {
                let s = if self.upsample { resize_bilinear_factor(s, 2.0)? } else { s.clone() };
                out + s
            }
            None => Ok(out),
        }
    }
}

pub(crate) struct ResBlock {
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
    pub(crate) skip: Conv2d,
    pub(crate) factor: f64,
}

impl ResBlock {
    fn load(w: &Weights, p: &str, factor: f64) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::load(w, &format!("{p}.conv1"), 1, 1)?,
            conv2: Conv2d::load(w, &format!("{p}.conv2"), 1, 1)?,
            skip: Conv2d::load(w, &format!("{p}.skip"), 1, 0)?,
            factor,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = leaky_relu(&self.conv1.forward(x)?, 0.2)?;
        let out = resize_bilinear_factor(&out, self.factor)?;
        let out = leaky_relu(&self.conv2.forward(&out)?, 0.2)?;
        let skip = self.skip.forward(&resize_bilinear_factor(x, self.factor)?)?;
        out + skip
    }
}

pub struct Gfpgan {
    pub(crate) conv_body_first: Conv2d,
    pub(crate) down: Vec<ResBlock>,
    pub(crate) final_conv: Conv2d,
    pub(crate) up: Vec<ResBlock>,
    pub(crate) final_linear: Linear,
    pub(crate) cond_scale: Vec<(Conv2d, Conv2d)>,
    pub(crate) cond_shift: Vec<(Conv2d, Conv2d)>,
    pub(crate) constant: Tensor,
    pub(crate) style_conv1: StyleConv,
    pub(crate) to_rgb1: ToRgb,
    pub(crate) style_convs: Vec<StyleConv>,
    pub(crate) to_rgbs: Vec<ToRgb>,
    pub(crate) num_style_feat: usize,
}

impl Gfpgan {
    /// Input/output face size.
    pub const SIZE: usize = 512;

    pub fn load(w: &Weights) -> Result<Self> {
        let n_down = w.count_indexed("conv_body_down.");
        let n_up = w.count_indexed("conv_body_up.");
        if n_down != 7 || n_up != 7 {
            candle_core::bail!("unsupported GFPGAN checkpoint (expected 512px v1.3/v1.4 layout)");
        }
        let noise = |i: usize| w.get(&format!("stylegan_decoder.noises.noise{i}"));
        let mut style_convs = Vec::new();
        let n_sc = w.count_indexed("stylegan_decoder.style_convs.");
        for i in 0..n_sc {
            style_convs.push(StyleConv::load(w, &format!("stylegan_decoder.style_convs.{i}"), i % 2 == 0, noise(i + 1)?)?);
        }
        let n_rgb = w.count_indexed("stylegan_decoder.to_rgbs.");
        let to_rgbs = (0..n_rgb).map(|i| ToRgb::load(w, &format!("stylegan_decoder.to_rgbs.{i}"), true)).collect::<Result<Vec<_>>>()?;
        let cond = |name: &str| -> Result<Vec<(Conv2d, Conv2d)>> {
            (0..n_up).map(|i| Ok((Conv2d::load(w, &format!("{name}.{i}.0"), 1, 1)?, Conv2d::load(w, &format!("{name}.{i}.2"), 1, 1)?))).collect()
        };
        let modw = w.get("stylegan_decoder.style_conv1.modulated_conv.modulation.weight")?;
        Ok(Self {
            conv_body_first: Conv2d::load(w, "conv_body_first", 1, 0)?,
            down: (0..n_down).map(|i| ResBlock::load(w, &format!("conv_body_down.{i}"), 0.5)).collect::<Result<_>>()?,
            final_conv: Conv2d::load(w, "final_conv", 1, 1)?,
            up: (0..n_up).map(|i| ResBlock::load(w, &format!("conv_body_up.{i}"), 2.0)).collect::<Result<_>>()?,
            final_linear: Linear::load(w, "final_linear")?,
            cond_scale: cond("condition_scale")?,
            cond_shift: cond("condition_shift")?,
            constant: w.get("stylegan_decoder.constant_input.weight")?,
            style_conv1: StyleConv::load(w, "stylegan_decoder.style_conv1", false, noise(0)?)?,
            to_rgb1: ToRgb::load(w, "stylegan_decoder.to_rgb1", false)?,
            style_convs,
            to_rgbs,
            num_style_feat: modw.dim(1)?,
        })
    }

    /// Restore an aligned 512x512 RGB face (1,3,512,512) in 0..1; returns the same shape in 0..1.
    pub fn forward(&self, face: &Tensor) -> Result<Tensor> {
        let x = ((face - 0.5)? / 0.5)?;
        let mut feat = leaky_relu(&self.conv_body_first.forward(&x)?, 0.2)?;
        let mut skips = Vec::new();
        for d in &self.down {
            feat = d.forward(&feat)?;
            skips.insert(0, feat.clone());
        }
        feat = leaky_relu(&self.final_conv.forward(&feat)?, 0.2)?;
        let style = self.final_linear.forward(&feat.flatten_all()?.unsqueeze(0)?)?;
        let n_latent = style.dim(1)? / self.num_style_feat;
        let latent = style.reshape((n_latent, self.num_style_feat))?;
        let mut conditions = Vec::new();
        for (i, u) in self.up.iter().enumerate() {
            feat = (feat + &skips[i])?;
            feat = u.forward(&feat)?;
            let (a, b) = &self.cond_scale[i];
            conditions.push(b.forward(&leaky_relu(&a.forward(&feat)?, 0.2)?)?);
            let (a, b) = &self.cond_shift[i];
            conditions.push(b.forward(&leaky_relu(&a.forward(&feat)?, 0.2)?)?);
        }
        let lat = |i: usize| latent.narrow(0, i, 1);
        let mut out = self.style_conv1.forward(&self.constant, &lat(0)?)?;
        let mut skip = self.to_rgb1.forward(&out, &lat(1)?, None)?;
        let mut i = 1;
        for (k, to_rgb) in self.to_rgbs.iter().enumerate() {
            out = self.style_convs[2 * k].forward(&out, &lat(i)?)?;
            if i < conditions.len() {
                let c = out.dim(1)? / 2;
                let same = out.narrow(1, 0, c)?;
                let sft = out.narrow(1, c, c)?;
                let sft = ((sft * &conditions[i - 1])? + &conditions[i])?;
                out = Tensor::cat(&[&same, &sft], 1)?;
            }
            out = self.style_convs[2 * k + 1].forward(&out, &lat(i + 1)?)?;
            skip = to_rgb.forward(&out, &lat(i + 2)?, Some(&skip))?;
            i += 2;
        }
        (skip.clamp(-1f32, 1f32)? + 1.0)? * 0.5
    }
}
