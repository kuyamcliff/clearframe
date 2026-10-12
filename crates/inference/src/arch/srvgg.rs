//! Real-ESRGAN `SRVGGNetCompact` (realesr-animevideov3, realesr-general-x4v3).

use candle_core::{Result, Tensor};

use crate::ops::{prelu, pixel_shuffle, upsample_nearest, Conv2d};
use crate::weights::Weights;

pub struct SrVggCompact {
    convs: Vec<Conv2d>,
    acts: Vec<Tensor>,
    pub scale: usize,
}

impl SrVggCompact {
    pub fn load(w: &Weights) -> Result<Self> {
        let n = w.count_indexed("body.");
        if n < 3 {
            candle_core::bail!("not an SRVGGNetCompact checkpoint");
        }
        let mut convs = Vec::new();
        let mut acts = Vec::new();
        for i in 0..n {
            if i % 2 == 0 {
                convs.push(Conv2d::load(w, &format!("body.{i}"), 1, 1)?);
            } else {
                acts.push(w.get(&format!("body.{i}.weight"))?);
            }
        }
        let out_ch = convs.last().unwrap().weight.dim(0)?;
        let scale = ((out_ch / 3) as f64).sqrt().round() as usize;
        Ok(Self { convs, acts, scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for (i, conv) in self.convs.iter().enumerate() {
            out = conv.forward(&out)?;
            if let Some(a) = self.acts.get(i) {
                out = prelu(&out, a)?;
            }
        }
        let out = pixel_shuffle(&out, self.scale)?;
        out + upsample_nearest(x, self.scale)?
    }
}
