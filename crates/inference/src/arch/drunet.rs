//! KAIR `UNetRes` (DRUNet) — noise-level-conditioned denoiser.

use candle_core::{Result, Tensor};

use crate::ops::{pad_replicate, Conv2d, ConvTranspose2d};
use crate::weights::Weights;

pub(crate) struct ResBlock {
    pub(crate) c1: Conv2d,
    pub(crate) c2: Conv2d,
}

impl ResBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self { c1: Conv2d::load(w, &format!("{p}.res.0"), 1, 1)?, c2: Conv2d::load(w, &format!("{p}.res.2"), 1, 1)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.c2.forward(&self.c1.forward_act(x, crate::ops::Act::Relu)?)?;
        x + r
    }
}

pub struct DruNet {
    pub(crate) head: Conv2d,
    pub(crate) downs: Vec<(Vec<ResBlock>, Conv2d)>,
    pub(crate) body: Vec<ResBlock>,
    pub(crate) ups: Vec<(ConvTranspose2d, Vec<ResBlock>)>,
    pub(crate) tail: Conv2d,
}

impl DruNet {
    pub fn load(w: &Weights) -> Result<Self> {
        let nb = 4;
        let blocks = |p: &str, off: usize| -> Result<Vec<ResBlock>> { (0..nb).map(|i| ResBlock::load(w, &format!("{p}.{}", i + off))).collect() };
        let mut downs = Vec::new();
        for d in 1..=3 {
            let p = format!("m_down{d}");
            downs.push((blocks(&p, 0)?, Conv2d::load(w, &format!("{p}.{nb}"), 2, 0)?));
        }
        let body = blocks("m_body", 0)?;
        let mut ups = Vec::new();
        for u in [3, 2, 1] {
            let p = format!("m_up{u}");
            ups.push((ConvTranspose2d::load(w, &format!("{p}.0"), 2, 0)?, blocks(&p, 1)?));
        }
        Ok(Self { head: Conv2d::load(w, "m_head", 1, 1)?, downs, body, ups, tail: Conv2d::load(w, "m_tail", 1, 1)? })
    }

    /// `sigma` on a 0..1 scale.
    pub fn forward(&self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let ph = (8 - h % 8) % 8;
        let pw = (8 - w % 8) % 8;
        let xp = pad_replicate(x, pw, ph)?;
        let (_, _, hh, ww) = xp.dims4()?;
        let m = Tensor::full(sigma, (1, 1, hh, ww), x.device())?.to_dtype(x.dtype())?;
        let x0 = Tensor::cat(&[&xp, &m], 1)?;
        let x1 = self.head.forward(&x0)?;
        let mut skips = vec![x1.clone()];
        let mut y = x1;
        for (blocks, down) in &self.downs {
            for b in blocks {
                y = b.forward(&y)?;
            }
            y = down.forward(&y)?;
            skips.push(y.clone());
        }
        for b in &self.body {
            y = b.forward(&y)?;
        }
        for (i, (up, blocks)) in self.ups.iter().enumerate() {
            let skip = &skips[3 - i];
            y = up.forward(&(y + skip)?)?;
            for b in blocks {
                y = b.forward(&y)?;
            }
        }
        let y = self.tail.forward(&(y + &skips[0])?)?;
        y.narrow(2, 0, h)?.narrow(3, 0, w)
    }
}
