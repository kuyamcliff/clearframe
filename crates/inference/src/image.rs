//! Interleaved RGB float images and tensor conversion.

use candle_core::{DType, Device, Result, Tensor};

/// An interleaved RGB image with `f32` samples nominally in `0.0..=1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height * 3] }
    }

    pub fn from_data(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height * 3, "image buffer size mismatch");
        Self { width, height, data }
    }

    /// Build from packed little-endian RGB48 bytes.
    pub fn from_rgb48le(width: usize, height: usize, bytes: &[u8]) -> Self {
        let data = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c) as f32 / 65535.0).collect();
        Self::from_data(width, height, data)
    }

    /// Build from packed RGB24 bytes.
    pub fn from_rgb24(width: usize, height: usize, bytes: &[u8]) -> Self {
        let data = bytes.iter().map(|b| *b as f32 / 255.0).collect();
        Self::from_data(width, height, data)
    }

    /// Write packed little-endian RGB48 bytes into `out` (cleared first).
    pub fn write_rgb48le(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.data.len() * 2);
        for v in &self.data {
            let q = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            out.extend_from_slice(&q.to_le_bytes());
        }
    }

    pub fn to_rgb24(&self) -> Vec<u8> {
        self.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect()
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[(y * self.width + x) * 3 + c]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, c: usize, v: f32) {
        self.data[(y * self.width + x) * 3 + c] = v;
    }

    /// Copy a rectangular region.
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> RgbImage {
        let mut out = RgbImage::new(w, h);
        for row in 0..h {
            let src = ((y + row) * self.width + x) * 3;
            let dst = row * w * 3;
            out.data[dst..dst + w * 3].copy_from_slice(&self.data[src..src + w * 3]);
        }
        out
    }

    /// Paste `src` at (x, y); `src` must fit.
    pub fn paste(&mut self, src: &RgbImage, x: usize, y: usize) {
        for row in 0..src.height {
            let s = row * src.width * 3;
            let d = ((y + row) * self.width + x) * 3;
            self.data[d..d + src.width * 3].copy_from_slice(&src.data[s..s + src.width * 3]);
        }
    }

    /// Convert to an NCHW tensor (1,3,H,W).
    pub fn to_tensor(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = Tensor::from_slice(&self.data, (self.height, self.width, 3), &Device::Cpu)?;
        t.permute((2, 0, 1))?.contiguous()?.unsqueeze(0)?.to_device(device)?.to_dtype(dtype)
    }

    /// Convert from an NCHW tensor (1,3,H,W).
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        let (_, c, h, w) = t.dims4()?;
        if c != 3 {
            candle_core::bail!("expected 3 channels, got {c}");
        }
        let hwc = t.squeeze(0)?.permute((1, 2, 0))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
        let data = hwc.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self { width: w, height: h, data })
    }

    /// Bilinear resample (used for previews, face pipeline and blending).
    pub fn resize_bilinear(&self, w: usize, h: usize) -> RgbImage {
        let mut out = RgbImage::new(w, h);
        let sx = self.width as f32 / w as f32;
        let sy = self.height as f32 / h as f32;
        for y in 0..h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (self.height - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = fy - y0 as f32;
            for x in 0..w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (self.width - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = fx - x0 as f32;
                for c in 0..3 {
                    let a = self.get(x0, y0, c) * (1.0 - wx) + self.get(x1, y0, c) * wx;
                    let b = self.get(x0, y1, c) * (1.0 - wx) + self.get(x1, y1, c) * wx;
                    out.set(x, y, c, a * (1.0 - wy) + b * wy);
                }
            }
        }
        out
    }

    /// Bicubic resample (Catmull-Rom, separable), sharper than bilinear when enlarging.
    /// Used as the plain-resize side of the strength blend.
    pub fn resize_bicubic(&self, w: usize, h: usize) -> RgbImage {
        fn taps(n_in: usize, n_out: usize) -> Vec<([usize; 4], [f32; 4])> {
            let s = n_in as f32 / n_out as f32;
            let k = |t: f32| {
                let t = t.abs();
                if t < 1.0 {
                    1.5 * t * t * t - 2.5 * t * t + 1.0
                } else if t < 2.0 {
                    -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
                } else {
                    0.0
                }
            };
            (0..n_out)
                .map(|o| {
                    let f = (o as f32 + 0.5) * s - 0.5;
                    let i0 = f.floor();
                    let mut idx = [0usize; 4];
                    let mut wt = [0f32; 4];
                    for j in 0..4 {
                        let i = i0 as isize - 1 + j as isize;
                        idx[j] = i.clamp(0, n_in as isize - 1) as usize;
                        wt[j] = k(f - (i0 + j as f32 - 1.0));
                    }
                    let sum: f32 = wt.iter().sum();
                    wt.iter_mut().for_each(|v| *v /= sum);
                    (idx, wt)
                })
                .collect()
        }
        let tx = taps(self.width, w);
        let ty = taps(self.height, h);
        // Horizontal pass, then vertical.
        let mut mid = RgbImage::new(w, self.height);
        for y in 0..self.height {
            for (x, (idx, wt)) in tx.iter().enumerate() {
                for c in 0..3 {
                    let v: f32 = (0..4).map(|j| self.get(idx[j], y, c) * wt[j]).sum();
                    mid.set(x, y, c, v);
                }
            }
        }
        let mut out = RgbImage::new(w, h);
        for (y, (idx, wt)) in ty.iter().enumerate() {
            for x in 0..w {
                for c in 0..3 {
                    let v: f32 = (0..4).map(|j| mid.get(x, idx[j], c) * wt[j]).sum();
                    out.set(x, y, c, v);
                }
            }
        }
        out
    }

    /// Linear blend: `self * (1 - t) + other * t`.
    pub fn blend(&self, other: &RgbImage, t: f32) -> RgbImage {
        assert_eq!((self.width, self.height), (other.width, other.height));
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a * (1.0 - t) + b * t).collect();
        RgbImage { width: self.width, height: self.height, data }
    }

    /// Mean absolute difference (used for scene-cut detection).
    pub fn mean_abs_diff(&self, other: &RgbImage) -> f32 {
        if self.data.len() != other.data.len() || self.data.is_empty() {
            return 1.0;
        }
        // Subsample for speed.
        let step = (self.data.len() / 30_000).max(1) * 3;
        let mut sum = 0.0f64;
        let mut n = 0usize;
        let mut i = 0;
        while i + 2 < self.data.len() {
            for c in 0..3 {
                sum += (self.data[i + c] - other.data[i + c]).abs() as f64;
            }
            n += 3;
            i += step;
        }
        (sum / n.max(1) as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_keeps_flat_areas_and_matches_bilinear_at_same_size() {
        let mut img = RgbImage::new(4, 3);
        img.data.iter_mut().for_each(|v| *v = 0.25);
        let up = img.resize_bicubic(16, 12);
        assert!(up.data.iter().all(|v| (v - 0.25).abs() < 1e-6));
        let mut ramp = RgbImage::new(8, 2);
        for x in 0..8 {
            for c in 0..3 {
                ramp.set(x, 0, c, x as f32 / 7.0);
                ramp.set(x, 1, c, x as f32 / 7.0);
            }
        }
        assert_eq!(ramp.resize_bicubic(8, 2), ramp);
    }

    #[test]
    fn tensor_roundtrip() {
        let mut img = RgbImage::new(5, 3);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = i as f32 / 45.0;
        }
        let t = img.to_tensor(&Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 3, 5]);
        let back = RgbImage::from_tensor(&t).unwrap();
        assert_eq!(img, back);
    }

    #[test]
    fn rgb48_roundtrip() {
        let img = RgbImage::from_data(1, 1, vec![0.0, 0.5, 1.0]);
        let mut bytes = Vec::new();
        img.write_rgb48le(&mut bytes);
        let back = RgbImage::from_rgb48le(1, 1, &bytes);
        for (a, b) in img.data.iter().zip(&back.data) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn crop_paste() {
        let mut img = RgbImage::new(4, 4);
        let patch = RgbImage::from_data(2, 1, vec![1.0; 6]);
        img.paste(&patch, 1, 2);
        assert_eq!(img.get(1, 2, 0), 1.0);
        assert_eq!(img.get(0, 2, 0), 0.0);
        assert_eq!(img.crop(1, 2, 2, 1), patch);
    }
}
