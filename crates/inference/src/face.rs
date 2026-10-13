//! Face detection, alignment, restoration and paste-back (GFPGAN pipeline).
//!
//! Mirrors facexlib's `FaceRestoreHelper` (square mask, no parsing network)
//! with two additions for video: landmark smoothing across frames and an
//! explicit identity-preserving `strength` blend.

use candle_core::{DType, Result};
use serde::{Deserialize, Serialize};

use crate::arch::gfpgan::Gfpgan;
use crate::arch::retinaface::Detection;
use crate::image::RgbImage;
use crate::model::FaceModels;

/// The 512x512 five-point template used by GFPGAN.
pub const FACE_TEMPLATE: [[f32; 2]; 5] = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]];

/// Grey border fill used by facexlib (BGR 135,133,132).
const BORDER_RGB: [f32; 3] = [132.0 / 255.0, 133.0 / 255.0, 135.0 / 255.0];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceParams {
    /// 0..1 blend between the aligned original and the restored face.
    pub strength: f32,
    /// Detection confidence threshold (facexlib uses 0.97).
    pub detection_threshold: f32,
    /// Maximum number of faces restored per frame (largest first).
    pub max_faces: usize,
    /// 0..0.95 exponential smoothing of landmarks across frames (reduces jitter).
    pub temporal_smoothing: f32,
    /// Longest side used for detection (frames are downscaled for speed).
    pub detection_max_side: usize,
}

impl Default for FaceParams {
    fn default() -> Self {
        Self { strength: 0.7, detection_threshold: 0.97, max_faces: 8, temporal_smoothing: 0.3, detection_max_side: 1280 }
    }
}

/// 2x3 affine matrix mapping (x, y) -> (a*x + b*y + c, d*x + e*y + f).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Affine {
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }
    pub fn invert(&self) -> Affine {
        let [a, b, c, d, e, f] = self.0;
        let det = a * e - b * d;
        let det = if det.abs() < 1e-12 { 1e-12 } else { det };
        let ia = e / det;
        let ib = -b / det;
        let id = -d / det;
        let ie = a / det;
        Affine([ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)])
    }
}

/// Least-squares similarity transform (rotation, uniform scale, translation)
/// mapping `src` points onto `dst` points (Umeyama without reflection).
pub fn estimate_similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Affine {
    let n = src.len() as f64;
    let (mut sx, mut sy, mut dx, mut dy) = (0f64, 0f64, 0f64, 0f64);
    for (s, d) in src.iter().zip(dst) {
        sx += s[0] as f64;
        sy += s[1] as f64;
        dx += d[0] as f64;
        dy += d[1] as f64;
    }
    let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);
    // Solve for a = s*cos, b = s*sin in: [x' ] = [a -b][x] + t
    let (mut num_a, mut num_b, mut den) = (0f64, 0f64, 0f64);
    for (s, d) in src.iter().zip(dst) {
        let (x, y) = (s[0] as f64 - sx, s[1] as f64 - sy);
        let (u, v) = (d[0] as f64 - dx, d[1] as f64 - dy);
        num_a += x * u + y * v;
        num_b += x * v - y * u;
        den += x * x + y * y;
    }
    let den = den.max(1e-12);
    let a = num_a / den;
    let b = num_b / den;
    let tx = dx - (a * sx - b * sy);
    let ty = dy - (b * sx + a * sy);
    Affine([a as f32, -b as f32, tx as f32, b as f32, a as f32, ty as f32])
}

#[inline]
fn sample_bilinear(img: &RgbImage, x: f32, y: f32, border: Option<[f32; 3]>) -> [f32; 3] {
    let w = img.width as f32;
    let h = img.height as f32;
    if let Some(b) = border {
        if x < -1.0 || y < -1.0 || x > w || y > h {
            return b;
        }
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let mut out = [0f32; 3];
    for (dy, wy) in [(0i32, 1.0 - fy), (1, fy)] {
        for (dx, wx) in [(0i32, 1.0 - fx), (1, fx)] {
            let wgt = wx * wy;
            if wgt == 0.0 {
                continue;
            }
            let px = x0 as i32 + dx;
            let py = y0 as i32 + dy;
            let v = if px < 0 || py < 0 || px >= img.width as i32 || py >= img.height as i32 {
                match border {
                    Some(b) => b,
                    None => {
                        let cx = px.clamp(0, img.width as i32 - 1) as usize;
                        let cy = py.clamp(0, img.height as i32 - 1) as usize;
                        [img.get(cx, cy, 0), img.get(cx, cy, 1), img.get(cx, cy, 2)]
                    }
                }
            } else {
                [img.get(px as usize, py as usize, 0), img.get(px as usize, py as usize, 1), img.get(px as usize, py as usize, 2)]
            };
            for c in 0..3 {
                out[c] += v[c] * wgt;
            }
        }
    }
    out
}

/// Warp `img` into a `size`x`size` crop using `m` (maps source -> crop).
pub fn warp_crop(img: &RgbImage, m: &Affine, size: usize) -> RgbImage {
    let inv = m.invert();
    let mut out = RgbImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let (sx, sy) = inv.apply(x as f32, y as f32);
            let v = sample_bilinear(img, sx, sy, Some(BORDER_RGB));
            for (c, value) in v.iter().enumerate().take(3) {
                out.set(x, y, c, *value);
            }
        }
    }
    out
}

fn erode(mask: &mut [f32], w: usize, h: usize, k: usize) {
    if k <= 1 {
        return;
    }
    let anchor = k / 2;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::MAX;
            for i in 0..k {
                let xx = x as isize + i as isize - anchor as isize;
                // Pixels outside the region count as the region border value (replicate);
                // our regions always include a zero margin.
                let xx = xx.clamp(0, w as isize - 1) as usize;
                m = m.min(mask[y * w + xx]);
            }
            tmp[y * w + x] = m;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::MAX;
            for i in 0..k {
                let yy = (y as isize + i as isize - anchor as isize).clamp(0, h as isize - 1) as usize;
                m = m.min(tmp[yy * w + x]);
            }
            mask[y * w + x] = m;
        }
    }
}

fn gaussian_blur(mask: &mut [f32], w: usize, h: usize, ksize: usize) {
    if ksize < 3 {
        return;
    }
    let sigma = 0.3 * ((ksize as f32 - 1.0) * 0.5 - 1.0) + 0.8;
    let r = ksize / 2;
    let kernel: Vec<f32> = (0..ksize).map(|i| (-((i as f32 - r as f32).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|v| v / sum).collect();
    let reflect = |i: isize, n: usize| -> usize {
        // OpenCV BORDER_REFLECT_101
        let n = n as isize;
        if n == 1 {
            return 0;
        }
        let mut i = i;
        while i < 0 || i >= n {
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * (n - 1) - i;
            }
        }
        i as usize
    };
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in kernel.iter().enumerate() {
                acc += kv * mask[y * w + reflect(x as isize + i as isize - r as isize, w)];
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, kv) in kernel.iter().enumerate() {
                acc += kv * tmp[reflect(y as isize + i as isize - r as isize, h) * w + x];
            }
            mask[y * w + x] = acc;
        }
    }
}

/// Paste a restored `face` back into `frame` using inverse affine `inv` (crop -> frame).
pub fn paste_face(frame: &mut RgbImage, face: &RgbImage, m: &Affine) {
    let inv = m.invert();
    let size = face.width as f32;
    // Bounding box of the face square in frame coordinates.
    let corners = [(0.0, 0.0), (size, 0.0), (0.0, size), (size, size)];
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (cx, cy) in corners {
        let (x, y) = inv.apply(cx, cy);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let margin = 4.0;
    let rx0 = (x0 - margin).floor().max(0.0) as usize;
    let ry0 = (y0 - margin).floor().max(0.0) as usize;
    let rx1 = ((x1 + margin).ceil() as usize).min(frame.width);
    let ry1 = ((y1 + margin).ceil() as usize).min(frame.height);
    if rx1 <= rx0 || ry1 <= ry0 {
        return;
    }
    let (rw, rh) = (rx1 - rx0, ry1 - ry0);
    // Warp the all-ones mask (bilinear) and the face.
    let mut mask = vec![0f32; rw * rh];
    let mut pasted = vec![[0f32; 3]; rw * rh];
    for y in 0..rh {
        for x in 0..rw {
            let (fx, fy) = m.apply((x + rx0) as f32, (y + ry0) as f32);
            // bilinear coverage of the [0, size-1] square
            let cov = |v: f32| -> f32 {
                if v < -1.0 || v > size {
                    0.0
                } else if v < 0.0 {
                    1.0 + v
                } else if v > size - 1.0 {
                    size - v
                } else {
                    1.0
                }
            };
            let mv = cov(fx) * cov(fy);
            mask[y * rw + x] = mv;
            if mv > 0.0 {
                pasted[y * rw + x] = sample_bilinear(face, fx, fy, Some([0.0; 3]));
            }
        }
    }
    erode(&mut mask, rw, rh, 2);
    let area: f32 = mask.iter().sum();
    let w_edge = (area.sqrt() as usize) / 20;
    let mut soft = mask.clone();
    erode(&mut soft, rw, rh, w_edge * 2);
    gaussian_blur(&mut soft, rw, rh, w_edge * 2 + 1);
    for y in 0..rh {
        for x in 0..rw {
            let i = y * rw + x;
            let a = soft[i];
            if a <= 0.0 {
                continue;
            }
            let er = mask[i];
            for (c, &p) in pasted[i].iter().enumerate().take(3) {
                let orig = frame.get(x + rx0, y + ry0, c);
                let face_v = p * er;
                frame.set(x + rx0, y + ry0, c, a * face_v + (1.0 - a) * orig);
            }
        }
    }
}

/// Stateful per-video face restorer.
pub struct FaceRestorer {
    models: FaceModels,
    prev: Vec<Detection>,
}

/// Result statistics for one frame.
#[derive(Debug, Clone, Default)]
pub struct FaceFrameStats {
    pub faces_detected: usize,
    pub faces_restored: usize,
}

impl FaceRestorer {
    pub fn new(models: FaceModels) -> Self {
        Self { models, prev: Vec::new() }
    }

    /// Forget tracking state (call at scene cuts / seeks).
    pub fn reset(&mut self) {
        self.prev.clear();
    }

    pub fn detect(&self, img: &RgbImage, p: &FaceParams) -> Result<Vec<Detection>> {
        let long = img.width.max(img.height);
        let scale = if long > p.detection_max_side { p.detection_max_side as f32 / long as f32 } else { 1.0 };
        let small;
        let src = if scale < 1.0 {
            small = img.resize_bilinear(((img.width as f32 * scale).round() as usize).max(1), ((img.height as f32 * scale).round() as usize).max(1));
            &small
        } else {
            img
        };
        let t = src.to_tensor(&self.models.device, DType::F32)?;
        let mut dets = self.models.detect(&t, p.detection_threshold, 0.4)?;
        let inv = 1.0 / scale;
        for d in &mut dets {
            for v in d.bbox.iter_mut() {
                *v *= inv;
            }
            for lm in d.landmarks.iter_mut() {
                lm[0] *= inv;
                lm[1] *= inv;
            }
        }
        // Drop side/tiny faces (facexlib eye_dist_threshold=5).
        dets.retain(|d| {
            let dx = d.landmarks[0][0] - d.landmarks[1][0];
            let dy = d.landmarks[0][1] - d.landmarks[1][1];
            (dx * dx + dy * dy).sqrt() >= 5.0
        });
        // Largest faces first.
        dets.sort_by(|a, b| {
            let aa = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
            let bb = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
            bb.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)
        });
        dets.truncate(p.max_faces);
        Ok(dets)
    }

    fn smooth(&self, dets: &mut [Detection], alpha: f32) {
        if alpha <= 0.0 || self.prev.is_empty() {
            return;
        }
        let alpha = alpha.min(0.95);
        for d in dets.iter_mut() {
            let best = self.prev.iter().map(|p| (iou(&p.bbox, &d.bbox), p)).filter(|(i, _)| *i > 0.3).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((_, p)) = best {
                for k in 0..5 {
                    for j in 0..2 {
                        d.landmarks[k][j] = alpha * p.landmarks[k][j] + (1.0 - alpha) * d.landmarks[k][j];
                    }
                }
            }
        }
    }

    /// Restore faces in `img` in place.
    pub fn process(&mut self, img: &mut RgbImage, p: &FaceParams) -> Result<FaceFrameStats> {
        let mut dets = self.detect(img, p)?;
        self.smooth(&mut dets, p.temporal_smoothing);
        let mut stats = FaceFrameStats { faces_detected: dets.len(), faces_restored: 0 };
        let source = img.clone();
        for d in &dets {
            let m = estimate_similarity(&d.landmarks, &FACE_TEMPLATE);
            let crop = warp_crop(&source, &m, Gfpgan::SIZE);
            let t = crop.to_tensor(&self.models.device, DType::F32)?;
            let restored = RgbImage::from_tensor(&self.models.restore(&t)?)?;
            let restored = if p.strength < 1.0 { crop.blend(&restored, p.strength.clamp(0.0, 1.0)) } else { restored };
            paste_face(img, &restored, &m);
            stats.faces_restored += 1;
        }
        self.prev = dets;
        Ok(stats)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let aa = (a[2] - a[0]) * (a[3] - a[1]);
    let bb = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (aa + bb - inter).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_recovers_known_transform() {
        let angle = 0.3f32;
        let s = 1.7f32;
        let m = Affine([s * angle.cos(), -s * angle.sin(), 12.0, s * angle.sin(), s * angle.cos(), -5.0]);
        let src: Vec<[f32; 2]> = FACE_TEMPLATE.to_vec();
        let dst: Vec<[f32; 2]> = src
            .iter()
            .map(|p| {
                let (x, y) = m.apply(p[0], p[1]);
                [x, y]
            })
            .collect();
        let est = estimate_similarity(&src, &dst);
        for (a, b) in est.0.iter().zip(m.0.iter()) {
            assert!((a - b).abs() < 1e-3, "{est:?} vs {m:?}");
        }
        let inv = est.invert();
        let (x, y) = inv.apply(dst[0][0], dst[0][1]);
        assert!((x - src[0][0]).abs() < 1e-2 && (y - src[0][1]).abs() < 1e-2);
    }

    #[test]
    fn warp_then_paste_identity_region() {
        let mut frame = RgbImage::new(64, 64);
        for v in frame.data.iter_mut() {
            *v = 0.25;
        }
        // Identity-ish transform: face crop of 32 px at offset (16, 16).
        let m = Affine([1.0, 0.0, -16.0, 0.0, 1.0, -16.0]);
        let face = RgbImage::from_data(32, 32, vec![0.75; 32 * 32 * 3]);
        paste_face(&mut frame, &face, &m);
        // centre gets the face, corners untouched
        assert!((frame.get(32, 32, 0) - 0.75).abs() < 1e-3);
        assert_eq!(frame.get(0, 0, 0), 0.25);
    }
}
