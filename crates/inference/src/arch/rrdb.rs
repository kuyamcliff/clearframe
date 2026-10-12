//! ESRGAN `RRDBNet` (RealESRGAN_x4plus, x4plus_anime_6B, x2plus).

use candle_core::{Result, Tensor};

use crate::ops::{pixel_unshuffle, upsample_nearest, Act, Conv2d};

const L: Act<'static> = Act::LeakyRelu(0.2);
use crate::weights::Weights;

struct Rdb {
    convs: [Conv2d; 5],
}

impl Rdb {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let c = |i: usize| Conv2d::load(w, &format!("{p}.conv{i}"), 1, 1);
        Ok(Self { convs: [c(1)?, c(2)?, c(3)?, c(4)?, c(5)?] })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.convs[0].forward_act(x, L)?;
        let x2 = self.convs[1].forward_act(&Tensor::cat(&[x, &x1], 1)?, L)?;
        let x3 = self.convs[2].forward_act(&Tensor::cat(&[x, &x1, &x2], 1)?, L)?;
        let x4 = self.convs[3].forward_act(&Tensor::cat(&[x, &x1, &x2, &x3], 1)?, L)?;
        let x5 = self.convs[4].forward(&Tensor::cat(&[x, &x1, &x2, &x3, &x4], 1)?)?;
        (x5 * 0.2)? + x
    }
}

pub struct RrdbNet {
    conv_first: Conv2d,
    body: Vec<[Rdb; 3]>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    pub scale: usize,
}

impl RrdbNet {
    pub fn load(w: &Weights) -> Result<Self> {
        let conv_first = Conv2d::load(w, "conv_first", 1, 1)?;
        let in_ch = conv_first.weight.dim(1)?;
        let scale = match in_ch {
            3 => 4,
            12 => 2,
            48 => 1,
            _ => candle_core::bail!("unsupported RRDBNet input channels {in_ch}"),
        };
        let n = w.count_indexed("body.");
        let mut body = Vec::with_capacity(n);
        for i in 0..n {
            body.push([Rdb::load(w, &format!("body.{i}.rdb1"))?, Rdb::load(w, &format!("body.{i}.rdb2"))?, Rdb::load(w, &format!("body.{i}.rdb3"))?]);
        }
        Ok(Self {
            conv_first,
            body,
            conv_body: Conv2d::load(w, "conv_body", 1, 1)?,
            conv_up1: Conv2d::load(w, "conv_up1", 1, 1)?,
            conv_up2: Conv2d::load(w, "conv_up2", 1, 1)?,
            conv_hr: Conv2d::load(w, "conv_hr", 1, 1)?,
            conv_last: Conv2d::load(w, "conv_last", 1, 1)?,
            scale,
        })
    }

    /// Input height/width must be divisible by `pad_multiple()`.
    pub fn pad_multiple(&self) -> usize {
        4 / self.scale
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = match self.scale {
            2 => pixel_unshuffle(x, 2)?,
            1 => pixel_unshuffle(x, 4)?,
            _ => x.clone(),
        };
        let feat = self.conv_first.forward(&feat)?;
        let mut b = feat.clone();
        for rrdb in &self.body {
            let o = rrdb[0].forward(&b)?;
            let o = rrdb[1].forward(&o)?;
            let o = rrdb[2].forward(&o)?;
            b = ((o * 0.2)? + &b)?;
        }
        let feat = (feat + self.conv_body.forward(&b)?)?;
        let feat = self.conv_up1.forward_act(&upsample_nearest(&feat, 2)?, L)?;
        let feat = self.conv_up2.forward_act(&upsample_nearest(&feat, 2)?, L)?;
        self.conv_last.forward(&self.conv_hr.forward_act(&feat, L)?)
    }
}
