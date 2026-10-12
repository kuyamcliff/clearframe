//! KAIR `FFDNet` (color) — noise-level-conditioned denoiser.

use candle_core::{Result, Tensor};

use crate::ops::{pad_replicate, pixel_shuffle, pixel_unshuffle, Conv2d};
use crate::weights::Weights;

pub struct FfdNet {
    convs: Vec<Conv2d>,
}

impl FfdNet {
    pub fn load(w: &Weights) -> Result<Self> {
        let n = w.count_indexed("model.");
        let mut convs = Vec::new();
        for i in (0..n).step_by(2) {
            convs.push(Conv2d::load(w, &format!("model.{i}"), 1, 1)?);
        }
        if convs.len() < 3 {
            candle_core::bail!("not an FFDNet checkpoint");
        }
        Ok(Self { convs })
    }

    /// `sigma` is the noise standard deviation on a 0..1 scale (e.g. 15/255).
    pub fn forward(&self, x: &Tensor, sigma: f32) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let xp = pad_replicate(x, w % 2, h % 2)?;
        let xd = pixel_unshuffle(&xp, 2)?;
        let (_, _, hh, ww) = xd.dims4()?;
        let m = Tensor::full(sigma, (1, 1, hh, ww), x.device())?.to_dtype(x.dtype())?;
        let mut y = Tensor::cat(&[&xd, &m], 1)?;
        let last = self.convs.len() - 1;
        for (i, c) in self.convs.iter().enumerate() {
            y = c.forward_act(&y, if i != last { crate::ops::Act::Relu } else { crate::ops::Act::None })?;
        }
        let y = pixel_shuffle(&y, 2)?;
        y.narrow(2, 0, h)?.narrow(3, 0, w)
    }
}
