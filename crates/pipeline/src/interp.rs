//! Frame-rate conversion / slow motion by interpolation.

use clearframe_inference::RgbImage;

/// Synthesises in-between frames.
pub trait Synthesizer: Send {
    /// Frames at each fraction `ts` (0 < t < 1) between `a` and `b`.
    fn synthesize(&mut self, a: &RgbImage, b: &RgbImage, ts: &[f32]) -> anyhow::Result<Vec<RgbImage>>;
}

/// Linear cross-fade.
pub struct BlendSynth;

impl Synthesizer for BlendSynth {
    fn synthesize(&mut self, a: &RgbImage, b: &RgbImage, ts: &[f32]) -> anyhow::Result<Vec<RgbImage>> {
        Ok(ts.iter().map(|t| a.blend(b, *t)).collect())
    }
}

/// Statistics reported after processing.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct InterpStats {
    pub synthesized: u64,
    pub copied: u64,
    pub scene_cuts: u64,
    pub duplicates: u64,
}

/// Retimes a stream: output frame `j` shows input position `j / ratio`,
/// where `ratio = out_fps * slow_motion / in_fps`.
pub struct Retimer {
    ratio: f64,
    scene_threshold: Option<f32>,
    duplicate_threshold: f32,
    prev: Option<(u64, RgbImage)>,
    next_out: Option<u64>,
    pub stats: InterpStats,
}

const EPS: f64 = 1e-6;

impl Retimer {
    pub fn new(ratio: f64, scene_threshold: Option<f32>, duplicate_threshold: f32) -> Self {
        assert!(ratio > 0.0);
        Self { ratio, scene_threshold, duplicate_threshold, prev: None, next_out: None, stats: InterpStats::default() }
    }

    fn pos(&self, j: u64) -> f64 {
        j as f64 / self.ratio
    }

    /// Feed input frame with absolute index `k`; returns (output index, frame) pairs.
    pub fn push(&mut self, k: u64, img: RgbImage, synth: &mut dyn Synthesizer) -> anyhow::Result<Vec<(u64, RgbImage)>> {
        let mut out = Vec::new();
        if self.next_out.is_none() {
            // First frame (possibly after a seek): first output whose position is >= k.
            self.next_out = Some(((k as f64) * self.ratio - EPS).ceil().max(0.0) as u64);
        }
        if let Some((pk, prev)) = self.prev.take() {
            if k != pk + 1 {
                anyhow::bail!("interpolation received frame {k} after {pk}; frames must be consecutive");
            }
            let mut js = Vec::new();
            let mut ts = Vec::new();
            let mut j = self.next_out.unwrap();
            while self.pos(j) < k as f64 - EPS {
                let t = (self.pos(j) - pk as f64).max(0.0);
                js.push(j);
                ts.push(t as f32);
                j += 1;
            }
            self.next_out = Some(j);
            if !js.is_empty() {
                let diff = if self.scene_threshold.is_some() || self.duplicate_threshold > 0.0 { prev.mean_abs_diff(&img) } else { 0.0 };
                let is_cut = self.scene_threshold.map(|th| diff > th).unwrap_or(false);
                let is_dup = self.duplicate_threshold > 0.0 && diff < self.duplicate_threshold;
                if is_cut {
                    self.stats.scene_cuts += 1;
                }
                if is_dup {
                    self.stats.duplicates += 1;
                }
                let need: Vec<(usize, f32)> = ts.iter().enumerate().filter(|(_, t)| **t > EPS as f32).map(|(i, t)| (i, *t)).collect();
                let synth_frames = if !need.is_empty() && !is_cut && !is_dup {
                    synth.synthesize(&prev, &img, &need.iter().map(|x| x.1).collect::<Vec<_>>())?
                } else {
                    Vec::new()
                };
                let mut si = 0;
                for (i, (jj, t)) in js.iter().zip(&ts).enumerate() {
                    let frame = if *t <= EPS as f32 {
                        self.stats.copied += 1;
                        prev.clone()
                    } else if is_cut || is_dup {
                        self.stats.copied += 1;
                        if *t < 0.5 || is_dup { prev.clone() } else { img.clone() }
                    } else {
                        debug_assert_eq!(need[si].0, i);
                        self.stats.synthesized += 1;
                        let f = synth_frames[si].clone();
                        si += 1;
                        f
                    };
                    out.push((*jj, frame));
                }
            }
        }
        self.prev = Some((k, img));
        Ok(out)
    }

    /// Flush: emit outputs up to `round((last + 1) * ratio)` using the last frame.
    pub fn finish(&mut self) -> Vec<(u64, RgbImage)> {
        let mut out = Vec::new();
        if let Some((k, img)) = self.prev.take() {
            let total = (((k + 1) as f64) * self.ratio).round() as u64;
            let mut j = self.next_out.unwrap_or(0);
            while j < total {
                out.push((j, img.clone()));
                self.stats.copied += 1;
                j += 1;
            }
            self.next_out = Some(j);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(v: f32) -> RgbImage {
        RgbImage::from_data(2, 2, vec![v; 12])
    }

    fn run(r: &mut Retimer, start: u64, n: u64) -> Vec<(u64, f32)> {
        let mut out = Vec::new();
        let mut s = BlendSynth;
        for k in start..start + n {
            for (j, f) in r.push(k, frame(k as f32 / 100.0), &mut s).unwrap() {
                out.push((j, f.data[0]));
            }
        }
        for (j, f) in r.finish() {
            out.push((j, f.data[0]));
        }
        out
    }

    #[test]
    fn doubles_frame_rate() {
        let mut r = Retimer::new(2.0, None, 0.0);
        let out = run(&mut r, 0, 4);
        let idx: Vec<u64> = out.iter().map(|x| x.0).collect();
        assert_eq!(idx, (0..8).collect::<Vec<_>>());
        assert!((out[1].1 - 0.005).abs() < 1e-6);
        assert_eq!(out[2].1, 0.01);
    }

    #[test]
    fn arbitrary_ratio_24_to_60() {
        let mut r = Retimer::new(2.5, None, 0.0);
        let out = run(&mut r, 0, 24);
        assert_eq!(out.len(), 60);
        assert!(out.windows(2).all(|w| w[1].0 == w[0].0 + 1));
    }

    #[test]
    fn slow_motion_4x() {
        let mut r = Retimer::new(4.0, None, 0.0);
        let out = run(&mut r, 0, 10);
        assert_eq!(out.len(), 40);
    }

    #[test]
    fn rate_reduction_drops_frames() {
        let mut r = Retimer::new(0.5, None, 0.0);
        let out = run(&mut r, 0, 10);
        assert_eq!(out.len(), 5);
        assert_eq!(out[1].1, 0.02);
    }

    #[test]
    fn resume_after_seek_is_consistent() {
        let mut full = Retimer::new(2.5, None, 0.0);
        let all = run(&mut full, 0, 30);
        let mut part = Retimer::new(2.5, None, 0.0);
        let tail = run(&mut part, 10, 20);
        // Every output produced after seeking matches the uninterrupted render.
        for (j, v) in &tail {
            let (_, w) = all.iter().find(|(a, _)| a == j).unwrap();
            assert!((v - w).abs() < 1e-6, "frame {j}");
        }
        assert_eq!(tail.first().unwrap().0, 25);
        assert_eq!(tail.last().unwrap().0, all.last().unwrap().0);
    }

    #[test]
    fn scene_cut_repeats_instead_of_blending() {
        let mut r = Retimer::new(2.0, Some(0.1), 0.0);
        let mut s = BlendSynth;
        r.push(0, frame(0.0), &mut s).unwrap();
        let out = r.push(1, frame(0.9), &mut s).unwrap();
        assert_eq!(out.len(), 2);
        // t = 0.5 at a cut shows the next shot instead of a cross-fade
        assert_eq!(out[1].1.data[0], 0.9);
        assert_eq!(r.stats.scene_cuts, 1);
    }

    #[test]
    fn non_consecutive_frames_error() {
        let mut r = Retimer::new(2.0, None, 0.0);
        let mut s = BlendSynth;
        r.push(0, frame(0.0), &mut s).unwrap();
        assert!(r.push(5, frame(0.0), &mut s).is_err());
    }
}
