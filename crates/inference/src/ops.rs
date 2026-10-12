//! Tensor building blocks shared by the model adapters.
//!
//! Everything here is expressed with backend-agnostic candle operations so that
//! the same code runs on CPU, CUDA and Metal builds. Tensors use NCHW layout
//! with a batch size of one.

use candle_core::{DType, Device, Module, Result, Tensor, D};

use crate::weights::Weights;

/// A 2D convolution with optional bias.
#[derive(Clone, Debug)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv2d {
    /// Load `{prefix}.weight` and optional `{prefix}.bias`.
    pub fn load(w: &Weights, prefix: &str, stride: usize, padding: usize) -> Result<Self> {
        Self::load_full(w, prefix, stride, padding, 1, 1)
    }

    pub fn load_full(
        w: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = w.get(&format!("{prefix}.weight"))?;
        let bias = w.opt(&format!("{prefix}.bias"));
        Ok(Self { weight, bias, stride, padding, dilation, groups })
    }

    /// Load a convolution followed by a BatchNorm (`{bn}.weight/bias/running_mean/running_var`)
    /// and fold the normalisation into the convolution weights.
    pub fn load_bn(w: &Weights, conv: &str, bn: &str, stride: usize, padding: usize, groups: usize) -> Result<Self> {
        let weight = w.get(&format!("{conv}.weight"))?;
        let gamma = w.get(&format!("{bn}.weight"))?;
        let beta = w.get(&format!("{bn}.bias"))?;
        let mean = w.get(&format!("{bn}.running_mean"))?;
        let var = w.get(&format!("{bn}.running_var"))?;
        let scale = (gamma / (var + 1e-5)?.sqrt()?)?;
        let out_c = weight.dim(0)?;
        let weight = weight.broadcast_mul(&scale.reshape((out_c, 1, 1, 1))?)?;
        let mut bias = (beta - (mean * &scale)?)?;
        if let Some(b) = w.opt(&format!("{conv}.bias")) {
            bias = (bias + (b * &scale)?)?;
        }
        Ok(Self { weight, bias: Some(bias), stride, padding, dilation: 1, groups })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = conv2d(x, &self.weight, self.padding, self.stride, self.dilation, self.groups)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
            None => Ok(y),
        }
    }
}

/// `Tensor::conv2d` with a workaround for a candle 0.9 CPU bug: when
/// `groups == 1` and channels == height == width, the tiled im2col kernel
/// mistakes an NCHW-contiguous input for NHWC and returns wrong values.
/// Appending one zero column avoids the ambiguous stride pattern without
/// changing the valid output region, which is then cropped back.
pub fn conv2d(x: &Tensor, w: &Tensor, padding: usize, stride: usize, dilation: usize, groups: usize) -> Result<Tensor> {
    let (_, c, h, wd) = x.dims4()?;
    if groups == 1 && c == h && h == wd && matches!(x.device(), Device::Cpu) {
        let (_, _, kh, kw) = w.dims4()?;
        let out_h = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
        let out_w = (wd + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
        let xp = x.pad_with_zeros(3, 0, 1)?;
        let y = xp.conv2d(w, padding, stride, dilation, groups)?;
        return y.narrow(2, 0, out_h)?.narrow(3, 0, out_w);
    }
    x.conv2d(w, padding, stride, dilation, groups)
}

/// A transposed 2D convolution (`groups == 1`).
#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl ConvTranspose2d {
    pub fn load(w: &Weights, prefix: &str, stride: usize, padding: usize) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.opt(&format!("{prefix}.bias")),
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose2d(&self.weight, self.padding, 0, self.stride, 1)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1))?),
            None => Ok(y),
        }
    }
}

/// Fully connected layer: `y = x W^T + b`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight_t: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self { weight_t: w.get(&format!("{prefix}.weight"))?.t()?.contiguous()?, bias: w.opt(&format!("{prefix}.bias")) })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight_t)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// LeakyReLU with slope < 1.
pub fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    x.maximum(&(x * slope)?)
}

/// Channel-wise PReLU with weights of shape `(C,)`.
pub fn prelu(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let c = weight.elem_count();
    let a = weight.reshape((1, c, 1, 1))?;
    let pos = x.relu()?;
    let neg = x.minimum(0f64)?;
    pos + neg.broadcast_mul(&a)?
}

/// `nn.PixelShuffle(r)`.
pub fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let oc = c / (r * r);
    x.reshape((b, oc, r, r, h, w))?.permute((0, 1, 4, 2, 5, 3))?.reshape((b, oc, h * r, w * r))
}

/// Inverse of [`pixel_shuffle`]; height and width must be divisible by `r`.
pub fn pixel_unshuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, hh, ww) = x.dims4()?;
    let (h, w) = (hh / r, ww / r);
    x.reshape((b, c, h, r, w, r))?.permute((0, 1, 3, 5, 2, 4))?.reshape((b, c * r * r, h, w))
}

/// Nearest-neighbour upsampling by an integer factor.
pub fn upsample_nearest(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    x.upsample_nearest2d(h * factor, w * factor)
}

/// Nearest-neighbour resize to an explicit size (PyTorch `mode='nearest'` semantics).
pub fn resize_nearest(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let dev = x.device();
    let ys: Vec<u32> = (0..oh).map(|i| ((i * h) / oh).min(h - 1) as u32).collect();
    let xs: Vec<u32> = (0..ow).map(|i| ((i * w) / ow).min(w - 1) as u32).collect();
    let x = x.index_select(&Tensor::new(ys.as_slice(), dev)?, 2)?;
    x.index_select(&Tensor::new(xs.as_slice(), dev)?, 3)
}

fn linear_taps(in_size: usize, out_size: usize, scale: f64, align_corners: bool) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut i0 = Vec::with_capacity(out_size);
    let mut i1 = Vec::with_capacity(out_size);
    let mut wt = Vec::with_capacity(out_size);
    for o in 0..out_size {
        let src = if align_corners {
            if out_size > 1 {
                o as f64 * (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
            } else {
                0.0
            }
        } else {
            ((o as f64 + 0.5) * scale - 0.5).max(0.0)
        };
        let lo = (src.floor() as usize).min(in_size - 1);
        let hi = (lo + 1).min(in_size - 1);
        let frac = (src - lo as f64) as f32;
        i0.push(lo as u32);
        i1.push(hi as u32);
        wt.push(if hi == lo { 0.0 } else { frac });
    }
    (i0, i1, wt)
}

/// Bilinear resize matching `F.interpolate(mode='bilinear', align_corners=False)`
/// when called with an explicit `scale_factor` (`src = (dst + 0.5) / factor - 0.5`).
pub fn resize_bilinear_factor(x: &Tensor, factor: f64) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let oh = ((h as f64) * factor).floor() as usize;
    let ow = ((w as f64) * factor).floor() as usize;
    resize_bilinear_impl(x, oh, ow, 1.0 / factor, 1.0 / factor, false)
}

/// Bilinear resize to an explicit size (`align_corners=False`, size-derived scale).
pub fn resize_bilinear(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    resize_bilinear_impl(x, oh, ow, h as f64 / oh as f64, w as f64 / ow as f64, false)
}

fn resize_bilinear_impl(x: &Tensor, oh: usize, ow: usize, sy: f64, sx: f64, align: bool) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    if oh == h && ow == w {
        return Ok(x.clone());
    }
    let dev = x.device();
    let dt = x.dtype();
    let (y0, y1, wy) = linear_taps(h, oh, sy, align);
    let a = x.index_select(&Tensor::new(y0.as_slice(), dev)?, 2)?;
    let b = x.index_select(&Tensor::new(y1.as_slice(), dev)?, 2)?;
    let wy = Tensor::new(wy.as_slice(), dev)?.to_dtype(dt)?.reshape((1, 1, oh, 1))?;
    let x = (a.broadcast_mul(&(1.0 - &wy)?)? + b.broadcast_mul(&wy)?)?;
    let (x0, x1, wx) = linear_taps(w, ow, sx, align);
    let a = x.index_select(&Tensor::new(x0.as_slice(), dev)?, 3)?;
    let b = x.index_select(&Tensor::new(x1.as_slice(), dev)?, 3)?;
    let wx = Tensor::new(wx.as_slice(), dev)?.to_dtype(dt)?.reshape((1, 1, 1, ow))?;
    a.broadcast_mul(&(1.0 - &wx)?)? + b.broadcast_mul(&wx)?
}

/// Backward warp of `img` (1,C,H,W) by `flow` (1,2,H,W) in pixels, matching
/// `grid_sample(mode='bilinear', padding_mode='border', align_corners=True)` as used by RIFE.
pub fn warp(img: &Tensor, flow: &Tensor) -> Result<Tensor> {
    let (_, c, h, w) = img.dims4()?;
    let dev = img.device();
    let dt = img.dtype();
    let gx = Tensor::arange(0u32, w as u32, dev)?.to_dtype(DType::F32)?.reshape((1, w))?.broadcast_as((h, w))?;
    let gy = Tensor::arange(0u32, h as u32, dev)?.to_dtype(DType::F32)?.reshape((h, 1))?.broadcast_as((h, w))?;
    let flow = flow.to_dtype(DType::F32)?;
    let fx = flow.get(0)?.get(0)?;
    let fy = flow.get(0)?.get(1)?;
    let sx = (gx + fx)?.clamp(0f32, (w - 1) as f32)?;
    let sy = (gy + fy)?.clamp(0f32, (h - 1) as f32)?;
    let x0 = sx.floor()?;
    let y0 = sy.floor()?;
    let wx = (&sx - &x0)?;
    let wy = (&sy - &y0)?;
    let x1 = (&x0 + 1.0)?.clamp(0f32, (w - 1) as f32)?;
    let y1 = (&y0 + 1.0)?.clamp(0f32, (h - 1) as f32)?;
    let wf = w as f64;
    let idx = |yy: &Tensor, xx: &Tensor| -> Result<Tensor> {
        ((yy * wf)? + xx)?.flatten_all()?.to_dtype(DType::U32)
    };
    let flat = img.reshape((c, h * w))?;
    let g = |i: Tensor| -> Result<Tensor> { flat.index_select(&i, 1)?.to_dtype(DType::F32) };
    let v00 = g(idx(&y0, &x0)?)?;
    let v01 = g(idx(&y0, &x1)?)?;
    let v10 = g(idx(&y1, &x0)?)?;
    let v11 = g(idx(&y1, &x1)?)?;
    let wx = wx.flatten_all()?.reshape((1, h * w))?;
    let wy = wy.flatten_all()?.reshape((1, h * w))?;
    let top = (v00.broadcast_mul(&(1.0 - &wx)?)? + v01.broadcast_mul(&wx)?)?;
    let bot = (v10.broadcast_mul(&(1.0 - &wx)?)? + v11.broadcast_mul(&wx)?)?;
    let out = (top.broadcast_mul(&(1.0 - &wy)?)? + bot.broadcast_mul(&wy)?)?;
    out.reshape((1, c, h, w))?.to_dtype(dt)
}

/// Pad right/bottom by replicating the edge pixels.
pub fn pad_replicate(x: &Tensor, right: usize, bottom: usize) -> Result<Tensor> {
    let mut x = x.clone();
    if right > 0 {
        let w = x.dim(3)?;
        let last = x.narrow(3, w - 1, 1)?;
        let rep = last.repeat((1, 1, 1, right))?;
        x = Tensor::cat(&[&x, &rep], 3)?;
    }
    if bottom > 0 {
        let h = x.dim(2)?;
        let last = x.narrow(2, h - 1, 1)?;
        let rep = last.repeat((1, 1, bottom, 1))?;
        x = Tensor::cat(&[&x, &rep], 2)?;
    }
    Ok(x)
}

/// Pad right/bottom with zeros.
pub fn pad_zero(x: &Tensor, right: usize, bottom: usize) -> Result<Tensor> {
    x.pad_with_zeros(3, 0, right)?.pad_with_zeros(2, 0, bottom)
}

/// Reflect padding on all sides (PyTorch `reflect`, edge not repeated).
pub fn pad_reflect(x: &Tensor, left: usize, right: usize, top: usize, bottom: usize) -> Result<Tensor> {
    let reflect_dim = |x: &Tensor, dim: usize, before: usize, after: usize| -> Result<Tensor> {
        if before == 0 && after == 0 {
            return Ok(x.clone());
        }
        let n = x.dim(dim)?;
        let dev = x.device();
        let mut idx: Vec<u32> = Vec::with_capacity(n + before + after);
        for i in (1..=before).rev() {
            idx.push(reflect_index(-(i as i64), n) as u32);
        }
        for i in 0..n {
            idx.push(i as u32);
        }
        for i in 0..after {
            idx.push(reflect_index((n + i) as i64, n) as u32);
        }
        x.index_select(&Tensor::new(idx.as_slice(), dev)?, dim)
    };
    let x = reflect_dim(x, 3, left, right)?;
    reflect_dim(&x, 2, top, bottom)
}

fn reflect_index(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    let mut m = i.rem_euclid(period);
    if m >= n as i64 {
        m = period - m;
    }
    m as usize
}

/// Channel LayerNorm used by NAFNet (`LayerNorm2d`).
pub fn layer_norm2d(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let c = weight.elem_count();
    let mu = x.mean_keepdim(1)?;
    let xc = x.broadcast_sub(&mu)?;
    let var = xc.sqr()?.mean_keepdim(1)?;
    let y = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    y.broadcast_mul(&weight.reshape((1, c, 1, 1))?)?.broadcast_add(&bias.reshape((1, c, 1, 1))?)
}

/// Global average pool keeping dims: (1,C,H,W) -> (1,C,1,1).
pub fn global_avg_pool(x: &Tensor) -> Result<Tensor> {
    x.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)
}

/// Softmax over the last dimension.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x)
}

/// Sequential helper for closures.
pub struct Seq(pub Vec<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>);

impl Module for Seq {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for f in &self.0 {
            x = f(&x)?;
        }
        Ok(x)
    }
}

/// Create a zero-initialised tensor helper for tests.
pub fn zeros(shape: &[usize], dev: &Device) -> Result<Tensor> {
    Tensor::zeros(shape, DType::F32, dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32], shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::new(v, &Device::Cpu).unwrap().reshape(shape).unwrap()
    }

    #[test]
    fn conv2d_square_channel_bug_is_worked_around() {
        // c == h == w triggers the upstream bug; compare with a non-square equivalent.
        let x = Tensor::randn(0f32, 1., (1, 24, 24, 24), &Device::Cpu).unwrap();
        let w = Tensor::randn(0f32, 0.1, (8, 24, 3, 3), &Device::Cpu).unwrap();
        let y = conv2d(&x, &w, 1, 1, 1, 1).unwrap();
        let x2 = Tensor::cat(&[&x, &x], 3).unwrap(); // 24x48: unaffected path
        let y2 = x2.conv2d(&w, 1, 1, 1, 1).unwrap().narrow(3, 0, 23).unwrap();
        let d = (y.narrow(3, 0, 23).unwrap() - y2).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-4, "diff {d}");
        assert_eq!(y.dims(), &[1, 8, 24, 24]);
    }

    #[test]
    fn shuffle_roundtrip() {
        let x = Tensor::arange(0f32, 64., &Device::Cpu).unwrap().reshape((1, 4, 4, 4)).unwrap();
        let y = pixel_unshuffle(&pixel_shuffle(&x, 2).unwrap(), 2).unwrap();
        let d = (x - y).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0);
    }

    #[test]
    fn pixel_shuffle_layout() {
        // channel c*r*r + i*r + j goes to (h*r+i, w*r+j)
        let x = t(&[1., 2., 3., 4.], (1, 4, 1, 1));
        let y = pixel_shuffle(&x, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![1., 2., 3., 4.]);
    }

    #[test]
    fn bilinear_identity_and_upscale() {
        let x = t(&[0., 1., 2., 3.], (1, 1, 2, 2));
        let y = resize_bilinear_factor(&x, 2.0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // PyTorch: [[0,.25,.75,1],[.5,.75,1.25,1.5],[1.5,1.75,2.25,2.5],[2,2.25,2.75,3]]
        let expect = [0., 0.25, 0.75, 1., 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2., 2.25, 2.75, 3.];
        for (a, b) in y.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{y:?}");
        }
    }

    #[test]
    fn warp_zero_flow_is_identity() {
        let x = Tensor::arange(0f32, 12., &Device::Cpu).unwrap().reshape((1, 1, 3, 4)).unwrap();
        let f = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let y = warp(&x, &f).unwrap();
        let d = (x - y).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(d < 1e-5);
    }

    #[test]
    fn warp_shift_and_border() {
        let x = t(&[0., 10., 20., 30.], (1, 1, 1, 4));
        let mut f = vec![1.0f32; 4];
        f.extend(vec![0.0f32; 4]);
        let f = Tensor::new(f.as_slice(), &Device::Cpu).unwrap().reshape((1, 2, 1, 4)).unwrap();
        let y = warp(&x, &f).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![10., 20., 30., 30.]);
    }

    #[test]
    fn reflect_pad() {
        let x = t(&[1., 2., 3.], (1, 1, 1, 3));
        let y = pad_reflect(&x, 2, 2, 0, 0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y, vec![3., 2., 1., 2., 3., 2., 1.]);
    }
}
