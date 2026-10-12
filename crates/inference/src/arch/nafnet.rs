//! NAFNet (deblur / denoise).

use candle_core::{Result, Tensor};

use crate::ops::{global_avg_pool, layer_norm2d, pad_zero, pixel_shuffle, Conv2d};
use crate::weights::Weights;

pub(crate) struct NafBlock {
    pub(crate) norm1: (Tensor, Tensor),
    pub(crate) norm2: (Tensor, Tensor),
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
    pub(crate) conv3: Conv2d,
    pub(crate) sca: Conv2d,
    pub(crate) conv4: Conv2d,
    pub(crate) conv5: Conv2d,
    pub(crate) beta: Tensor,
    pub(crate) gamma: Tensor,
}

fn gate(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)? / 2;
    x.narrow(1, 0, c)? * x.narrow(1, c, c)?
}

impl NafBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let conv2_w = w.get(&format!("{p}.conv2.weight"))?;
        let groups = conv2_w.dim(0)?;
        Ok(Self {
            norm1: (w.get(&format!("{p}.norm1.weight"))?, w.get(&format!("{p}.norm1.bias"))?),
            norm2: (w.get(&format!("{p}.norm2.weight"))?, w.get(&format!("{p}.norm2.bias"))?),
            conv1: Conv2d::load(w, &format!("{p}.conv1"), 1, 0)?,
            conv2: Conv2d::load_full(w, &format!("{p}.conv2"), 1, 1, 1, groups)?,
            conv3: Conv2d::load(w, &format!("{p}.conv3"), 1, 0)?,
            sca: Conv2d::load(w, &format!("{p}.sca.1"), 1, 0)?,
            conv4: Conv2d::load(w, &format!("{p}.conv4"), 1, 0)?,
            conv5: Conv2d::load(w, &format!("{p}.conv5"), 1, 0)?,
            beta: w.get(&format!("{p}.beta"))?,
            gamma: w.get(&format!("{p}.gamma"))?,
        })
    }

    fn forward(&self, inp: &Tensor) -> Result<Tensor> {
        let x = layer_norm2d(inp, &self.norm1.0, &self.norm1.1, 1e-6)?;
        let x = self.conv2.forward(&self.conv1.forward(&x)?)?;
        let x = gate(&x)?;
        let att = self.sca.forward(&global_avg_pool(&x)?)?;
        let x = x.broadcast_mul(&att)?;
        let x = self.conv3.forward(&x)?;
        let y = (inp + x.broadcast_mul(&self.beta)?)?;
        let x = self.conv4.forward(&layer_norm2d(&y, &self.norm2.0, &self.norm2.1, 1e-6)?)?;
        let x = self.conv5.forward(&gate(&x)?)?;
        y + x.broadcast_mul(&self.gamma)?
    }
}

pub struct NafNet {
    pub(crate) intro: Conv2d,
    pub(crate) ending: Conv2d,
    pub(crate) encoders: Vec<Vec<NafBlock>>,
    pub(crate) downs: Vec<Conv2d>,
    pub(crate) middle: Vec<NafBlock>,
    pub(crate) ups: Vec<Conv2d>,
    pub(crate) decoders: Vec<Vec<NafBlock>>,
}

impl NafNet {
    pub fn load(w: &Weights) -> Result<Self> {
        let n_enc = w.count_indexed("encoders.");
        let n_dec = w.count_indexed("decoders.");
        let blocks = |p: String| -> Result<Vec<NafBlock>> {
            let n = w.count_indexed(&format!("{p}."));
            (0..n).map(|i| NafBlock::load(w, &format!("{p}.{i}"))).collect()
        };
        let mut encoders = Vec::new();
        let mut downs = Vec::new();
        for i in 0..n_enc {
            encoders.push(blocks(format!("encoders.{i}"))?);
            downs.push(Conv2d::load(w, &format!("downs.{i}"), 2, 0)?);
        }
        let mut ups = Vec::new();
        let mut decoders = Vec::new();
        for i in 0..n_dec {
            ups.push(Conv2d::load(w, &format!("ups.{i}.0"), 1, 0)?);
            decoders.push(blocks(format!("decoders.{i}"))?);
        }
        Ok(Self { intro: Conv2d::load(w, "intro", 1, 1)?, ending: Conv2d::load(w, "ending", 1, 1)?, encoders, downs, middle: blocks("middle_blks".into())?, ups, decoders })
    }

    pub fn forward(&self, inp: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = inp.dims4()?;
        let m = 1usize << self.encoders.len();
        let inp_p = pad_zero(inp, (m - w % m) % m, (m - h % m) % m)?;
        let mut x = self.intro.forward(&inp_p)?;
        let mut encs = Vec::new();
        for (enc, down) in self.encoders.iter().zip(&self.downs) {
            for b in enc {
                x = b.forward(&x)?;
            }
            encs.push(x.clone());
            x = down.forward(&x)?;
        }
        for b in &self.middle {
            x = b.forward(&x)?;
        }
        for ((dec, up), skip) in self.decoders.iter().zip(&self.ups).zip(encs.iter().rev()) {
            x = pixel_shuffle(&up.forward(&x)?, 2)?;
            x = (x + skip)?;
            for b in dec {
                x = b.forward(&x)?;
            }
        }
        let x = (self.ending.forward(&x)? + inp_p)?;
        x.narrow(2, 0, h)?.narrow(3, 0, w)
    }
}
