//! Source noise estimate, used to pick the Real-ESRGAN General v3 denoise automatically.
//!
//! Clean or compressed video looks best with little denoise (it keeps skin, hair and fabric
//! texture); grainy video needs more. Measured on YouTube-style 240p-720p encodes against
//! the 1080p original: clean clips (noise 0.1-0.9) score best at 0%, a grainy clip (1.7+)
//! at about 50%.

use std::process::Stdio;

use clearframe_media::probe::MediaInfo;
use clearframe_media::tools::FfmpegTools;

use crate::spec::TimeRange;

/// Frames sampled across the clip.
const SAMPLES: usize = 5;

/// Noise standard deviation (8-bit luma levels) of the flatter half of a grayscale image:
/// Immerkær's Laplacian-difference estimator, restricted to pixels whose Sobel gradient is at
/// or below the median so edges and texture are not counted as noise.
pub fn sigma_gray(width: usize, height: usize, px: &[u8]) -> Option<f32> {
    if width < 8 || height < 8 || px.len() < width * height {
        return None;
    }
    let at = |x: usize, y: usize| px[y * width + x] as f32;
    let n = (width - 2) * (height - 2);
    let mut lap = Vec::with_capacity(n);
    let mut grad = Vec::with_capacity(n);
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let l = at(x - 1, y - 1) - 2.0 * at(x, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y) + 4.0 * at(x, y) - 2.0 * at(x + 1, y) + at(x - 1, y + 1) - 2.0 * at(x, y + 1)
                + at(x + 1, y + 1);
            let gx = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
            let gy = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
            lap.push(l.abs());
            grad.push(gx.hypot(gy));
        }
    }
    let mut sorted = grad.clone();
    let mid = sorted.len() / 2;
    let (_, median, _) = sorted.select_nth_unstable_by(mid, f32::total_cmp);
    let thr = *median;
    let (sum, count) = lap.iter().zip(&grad).filter(|(_, g)| **g <= thr).fold((0f64, 0usize), |(s, c), (l, _)| (s + *l as f64, c + 1));
    if count == 0 {
        return None;
    }
    Some(((std::f64::consts::PI / 2.0).sqrt() / 6.0 * sum / count as f64) as f32)
}

/// General v3 denoise (0..1) for a measured noise level.
pub fn general_denoise_for(sigma: f32) -> f32 {
    ((sigma - 1.0) / 1.5).clamp(0.0, 1.0) * 0.6
}

/// Median noise level over a few frames spread across `range` (or the whole source).
/// `None` when no frame could be read.
pub fn estimate(tools: &FfmpegTools, source: &std::path::Path, info: &MediaInfo, range: Option<TimeRange>) -> Option<f32> {
    let v = info.video().ok()?;
    let (w, h) = (v.width as usize, v.height as usize);
    let (start, end) = match range {
        Some(r) => (r.start, r.end),
        None => (0.0, info.duration),
    };
    let len = (end - start).max(0.0);
    let mut values: Vec<f32> = (0..SAMPLES)
        .filter_map(|i| {
            let t = start + len * (i as f64 + 0.5) / SAMPLES as f64;
            let mut c = tools.ffmpeg_cmd();
            c.args(["-v", "error", "-ss", &format!("{t:.3}")]).arg("-i").arg(source);
            c.args(["-map", "0:v:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]);
            c.stdin(Stdio::null()).stderr(Stdio::null());
            clearframe_media::tools::hide_console(&mut c);
            let out = c.output().ok()?;
            sigma_gray(w, h, &out.stdout)
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random noise in -1..1.
    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % 20001) as f32 / 10000.0 - 1.0
    }

    #[test]
    fn flat_image_is_clean_and_noise_raises_the_estimate() {
        let (w, h) = (96, 64);
        let flat = vec![120u8; w * h];
        assert!(sigma_gray(w, h, &flat).unwrap() < 0.01);
        let mut seed = 7u64;
        // Uniform noise of amplitude a has sigma a/sqrt(3).
        let noisy: Vec<u8> = (0..w * h).map(|_| (120.0 + 10.0 * rnd(&mut seed)).round() as u8).collect();
        let s = sigma_gray(w, h, &noisy).unwrap();
        assert!((3.5..8.0).contains(&s), "sigma {s}");
    }

    #[test]
    fn edges_do_not_count_as_noise() {
        let (w, h) = (96, 64);
        let stripes: Vec<u8> = (0..w * h).map(|i| if (i % w) / 8 % 2 == 0 { 30 } else { 220 }).collect();
        assert!(sigma_gray(w, h, &stripes).unwrap() < 0.5);
    }

    #[test]
    fn denoise_mapping() {
        assert_eq!(general_denoise_for(0.4), 0.0);
        assert_eq!(general_denoise_for(1.0), 0.0);
        assert!((general_denoise_for(1.75) - 0.3).abs() < 1e-6);
        assert_eq!(general_denoise_for(9.0), 0.6);
    }
}
