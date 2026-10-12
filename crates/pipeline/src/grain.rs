//! Deterministic procedural film grain.
//!
//! The pattern depends only on (seed, frame index, pixel position), so renders
//! are reproducible regardless of thread scheduling or resume points.

use clearframe_inference::RgbImage;
use rayon::prelude::*;

use crate::spec::GrainSettings;

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn uniform(h: u64) -> f32 {
    ((h >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}

/// Standard normal sample from a hash (Box-Muller).
#[inline]
fn gauss(h: u64) -> f32 {
    let u1 = uniform(h);
    let u2 = uniform(mix64(h ^ 0xD1B5_4A32_D192_ED03));
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Generate a noise plane of `w x h` at grain `size` (>= 1 means larger grains).
fn noise_plane(w: usize, h: usize, size: f32, key: u64) -> Vec<f32> {
    let size = size.max(1.0);
    let gw = ((w as f32 / size).ceil() as usize).max(1) + 1;
    let gh = ((h as f32 / size).ceil() as usize).max(1) + 1;
    let grid: Vec<f32> = (0..gw * gh).into_par_iter().map(|i| gauss(mix64(key ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))).collect();
    if size <= 1.0 {
        return (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| grid[y * gw + x]).collect();
    }
    // Bilinear upsampling of the coarse grid; renormalise variance lost by interpolation.
    let norm = 1.0 / 0.75f32.sqrt();
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = y as f32 / size;
        let y0 = fy.floor() as usize;
        let ty = fy - y0 as f32;
        for (x, v) in row.iter_mut().enumerate() {
            let fx = x as f32 / size;
            let x0 = fx.floor() as usize;
            let tx = fx - x0 as f32;
            let a = grid[y0 * gw + x0] * (1.0 - tx) + grid[y0 * gw + x0 + 1] * tx;
            let b = grid[(y0 + 1) * gw + x0] * (1.0 - tx) + grid[(y0 + 1) * gw + x0 + 1] * tx;
            *v = (a * (1.0 - ty) + b * ty) * norm;
        }
    });
    out
}

/// Apply grain in place.
pub fn apply_grain(img: &mut RgbImage, s: &GrainSettings, frame_index: u64) {
    if s.strength <= 0.0 {
        return;
    }
    let (w, h) = (img.width, img.height);
    let frame_key = if s.animated { frame_index } else { 0 };
    let base = mix64(s.seed ^ mix64(frame_key.wrapping_add(0x51_7CC1_B727_220A)));
    let mono = noise_plane(w, h, s.size, base);
    let color = s.color.clamp(0.0, 1.0);
    let chans: Option<[Vec<f32>; 3]> = if color > 0.0 {
        Some([noise_plane(w, h, s.size, mix64(base ^ 1)), noise_plane(w, h, s.size, mix64(base ^ 2)), noise_plane(w, h, s.size, mix64(base ^ 3))])
    } else {
        None
    };
    let norm = 1.0 / ((1.0 - color).powi(2) + color.powi(2)).sqrt();
    let amp = s.strength * 0.08;
    let luma_response = s.luma_response;
    img.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let p = &mut row[x * 3..x * 3 + 3];
            let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            let resp = if luma_response { 0.3 + 0.7 * (4.0 * l * (1.0 - l)).clamp(0.0, 1.0) } else { 1.0 };
            for c in 0..3 {
                let n = match &chans {
                    Some(ch) => (mono[i] * (1.0 - color) + ch[c][i] * color) * norm,
                    None => mono[i],
                };
                p[c] = (p[c] + n * amp * resp).clamp(0.0, 1.0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(w: usize, h: usize) -> RgbImage {
        RgbImage::from_data(w, h, vec![0.5; w * h * 3])
    }

    #[test]
    fn deterministic_and_seeded() {
        let s = GrainSettings::default();
        let mut a = grey(64, 48);
        let mut b = grey(64, 48);
        apply_grain(&mut a, &s, 7);
        apply_grain(&mut b, &s, 7);
        assert_eq!(a, b);
        let mut c = grey(64, 48);
        apply_grain(&mut c, &GrainSettings { seed: 2, ..s.clone() }, 7);
        assert_ne!(a, c);
    }

    #[test]
    fn animated_vs_static() {
        let s = GrainSettings { animated: false, ..Default::default() };
        let mut a = grey(32, 32);
        let mut b = grey(32, 32);
        apply_grain(&mut a, &s, 1);
        apply_grain(&mut b, &s, 2);
        assert_eq!(a, b);
        let s = GrainSettings { animated: true, ..Default::default() };
        let mut a = grey(32, 32);
        let mut b = grey(32, 32);
        apply_grain(&mut a, &s, 1);
        apply_grain(&mut b, &s, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn amplitude_and_mono() {
        let s = GrainSettings { strength: 0.5, color: 0.0, luma_response: false, size: 1.0, ..Default::default() };
        let mut a = grey(128, 128);
        apply_grain(&mut a, &s, 0);
        let mean: f32 = a.data.iter().sum::<f32>() / a.data.len() as f32;
        let var: f32 = a.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.data.len() as f32;
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
        assert!((var.sqrt() - 0.04).abs() < 0.006, "std {}", var.sqrt());
        // monochrome: channels identical per pixel
        assert!(a.data.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]));
        let off = GrainSettings { strength: 0.0, ..Default::default() };
        let mut b = grey(8, 8);
        apply_grain(&mut b, &off, 0);
        assert_eq!(b, grey(8, 8));
    }

    #[test]
    fn large_grain_size() {
        let s = GrainSettings { size: 3.0, ..Default::default() };
        let mut a = grey(50, 37);
        apply_grain(&mut a, &s, 0);
        assert!(a.data.iter().all(|v| v.is_finite()));
    }
}
