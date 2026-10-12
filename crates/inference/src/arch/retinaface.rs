//! RetinaFace face detector (facexlib weights; ResNet-50 or MobileNet-0.25 backbone).

use candle_core::{Result, Tensor};

use crate::ops::{resize_nearest, softmax_last, Conv2d};
use crate::weights::Weights;

/// Conv + folded BN + optional leaky ReLU.
struct Cbr {
    conv: Conv2d,
    act: Option<f64>,
}

impl Cbr {
    fn load(w: &Weights, conv: &str, bn: &str, stride: usize, pad: usize, groups: usize, act: Option<f64>) -> Result<Self> {
        Ok(Self { conv: Conv2d::load_bn(w, conv, bn, stride, pad, groups)?, act })
    }
    /// facexlib `Sequential(conv, bn[, act])` layout: `{p}.0` conv, `{p}.1` bn.
    fn seq(w: &Weights, p: &str, stride: usize, pad: usize, groups: usize, act: Option<f64>) -> Result<Self> {
        Self::load(w, &format!("{p}.0"), &format!("{p}.1"), stride, pad, groups, act)
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward_act(
            x,
            match self.act {
                Some(0.0) => crate::ops::Act::Relu,
                Some(s) => crate::ops::Act::LeakyRelu(s as f32),
                None => crate::ops::Act::None,
            },
        )
    }
}

struct Bottleneck {
    c1: Cbr,
    c2: Cbr,
    c3: Cbr,
    down: Option<Cbr>,
}

impl Bottleneck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.c3.forward(&self.c2.forward(&self.c1.forward(x)?)?)?;
        let id = match &self.down {
            Some(d) => d.forward(x)?,
            None => x.clone(),
        };
        (y + id)?.relu()
    }
}

enum Backbone {
    ResNet50 { stem: Cbr, layers: Vec<Vec<Bottleneck>> },
    MobileNet { stages: Vec<Vec<Cbr>> },
}

impl Backbone {
    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        match self {
            Backbone::ResNet50 { stem, layers } => {
                let mut y = stem.forward(x)?;
                y = max_pool_3x3_s2(&y)?;
                let mut outs = Vec::new();
                for (i, layer) in layers.iter().enumerate() {
                    for b in layer {
                        y = b.forward(&y)?;
                    }
                    if i >= 1 {
                        outs.push(y.clone());
                    }
                }
                Ok(outs)
            }
            Backbone::MobileNet { stages } => {
                let mut y = x.clone();
                let mut outs = Vec::new();
                for st in stages {
                    for c in st {
                        y = c.forward(&y)?;
                    }
                    outs.push(y.clone());
                }
                Ok(outs)
            }
        }
    }
}

/// 3x3 / stride 2 / padding 1 max pool (torchvision ResNet stem). The input is
/// post-ReLU (non-negative) so zero padding is equivalent to -inf padding.
fn max_pool_3x3_s2(x: &Tensor) -> Result<Tensor> {
    let x = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
    x.max_pool2d_with_stride(3, 2)
}

struct Ssh {
    c3: Cbr,
    c5_1: Cbr,
    c5_2: Cbr,
    c7_2: Cbr,
    c7_3: Cbr,
}

impl Ssh {
    fn load(w: &Weights, p: &str, leaky: f64) -> Result<Self> {
        Ok(Self {
            c3: Cbr::seq(w, &format!("{p}.conv3X3"), 1, 1, 1, None)?,
            c5_1: Cbr::seq(w, &format!("{p}.conv5X5_1"), 1, 1, 1, Some(leaky))?,
            c5_2: Cbr::seq(w, &format!("{p}.conv5X5_2"), 1, 1, 1, None)?,
            c7_2: Cbr::seq(w, &format!("{p}.conv7X7_2"), 1, 1, 1, Some(leaky))?,
            c7_3: Cbr::seq(w, &format!("{p}.conv7x7_3"), 1, 1, 1, None)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.c3.forward(x)?;
        let b1 = self.c5_1.forward(x)?;
        let b = self.c5_2.forward(&b1)?;
        let c = self.c7_3.forward(&self.c7_2.forward(&b1)?)?;
        Tensor::cat(&[&a, &b, &c], 1)?.relu()
    }
}

pub struct RetinaFace {
    backbone: Backbone,
    fpn_out: [Cbr; 3],
    merge1: Cbr,
    merge2: Cbr,
    ssh: [Ssh; 3],
    class_head: [Conv2d; 3],
    bbox_head: [Conv2d; 3],
    landmark_head: [Conv2d; 3],
}

/// A detected face in input-image pixel coordinates.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    /// Five landmarks: left eye, right eye, nose, left mouth corner, right mouth corner.
    pub landmarks: [[f32; 2]; 5],
}

impl RetinaFace {
    pub fn load(w: &Weights) -> Result<Self> {
        let resnet = w.contains("body.conv1.weight");
        let (backbone, leaky) = if resnet {
            let stem = Cbr::load(w, "body.conv1", "body.bn1", 2, 3, 1, Some(0.0))?;
            let mut layers = Vec::new();
            for (li, (n, stride)) in [(3usize, 1usize), (4, 2), (6, 2), (3, 2)].iter().enumerate() {
                let mut blocks = Vec::new();
                for bi in 0..*n {
                    let p = format!("body.layer{}.{bi}", li + 1);
                    let s = if bi == 0 { *stride } else { 1 };
                    let down = if bi == 0 { Some(Cbr::load(w, &format!("{p}.downsample.0"), &format!("{p}.downsample.1"), s, 0, 1, None)?) } else { None };
                    blocks.push(Bottleneck {
                        c1: Cbr::load(w, &format!("{p}.conv1"), &format!("{p}.bn1"), 1, 0, 1, Some(0.0))?,
                        c2: Cbr::load(w, &format!("{p}.conv2"), &format!("{p}.bn2"), s, 1, 1, Some(0.0))?,
                        c3: Cbr::load(w, &format!("{p}.conv3"), &format!("{p}.bn3"), 1, 0, 1, None)?,
                        down,
                    });
                }
                layers.push(blocks);
            }
            (Backbone::ResNet50 { stem, layers }, 0.0)
        } else if w.contains("body.stage1.0.0.weight") {
            // MobileNetV1 0.25
            let conv_dw = |p: &str, inp: usize, stride: usize| -> Result<Vec<Cbr>> {
                Ok(vec![Cbr::load(w, &format!("{p}.0"), &format!("{p}.1"), stride, 1, inp, Some(0.1))?, Cbr::load(w, &format!("{p}.3"), &format!("{p}.4"), 1, 0, 1, Some(0.1))?])
            };
            let mut s1 = vec![Cbr::seq(w, "body.stage1.0", 2, 1, 1, Some(0.1))?];
            for (i, (inp, s)) in [(8, 1), (16, 2), (32, 1), (32, 2), (64, 1)].iter().enumerate() {
                s1.extend(conv_dw(&format!("body.stage1.{}", i + 1), *inp, *s)?);
            }
            let mut s2 = Vec::new();
            for (i, (inp, s)) in [(64, 2), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1)].iter().enumerate() {
                s2.extend(conv_dw(&format!("body.stage2.{i}"), *inp, *s)?);
            }
            let mut s3 = Vec::new();
            for (i, (inp, s)) in [(128, 2), (256, 1)].iter().enumerate() {
                s3.extend(conv_dw(&format!("body.stage3.{i}"), *inp, *s)?);
            }
            (Backbone::MobileNet { stages: vec![s1, s2, s3] }, 0.1)
        } else {
            candle_core::bail!("unrecognised RetinaFace backbone");
        };
        let head = |name: &str, i: usize| Conv2d::load(w, &format!("{name}.{i}.conv1x1"), 1, 0);
        Ok(Self {
            backbone,
            fpn_out: [Cbr::seq(w, "fpn.output1", 1, 0, 1, Some(leaky))?, Cbr::seq(w, "fpn.output2", 1, 0, 1, Some(leaky))?, Cbr::seq(w, "fpn.output3", 1, 0, 1, Some(leaky))?],
            merge1: Cbr::seq(w, "fpn.merge1", 1, 1, 1, Some(leaky))?,
            merge2: Cbr::seq(w, "fpn.merge2", 1, 1, 1, Some(leaky))?,
            ssh: [Ssh::load(w, "ssh1", leaky)?, Ssh::load(w, "ssh2", leaky)?, Ssh::load(w, "ssh3", leaky)?],
            class_head: [head("ClassHead", 0)?, head("ClassHead", 1)?, head("ClassHead", 2)?],
            bbox_head: [head("BboxHead", 0)?, head("BboxHead", 1)?, head("BboxHead", 2)?],
            landmark_head: [head("LandmarkHead", 0)?, head("LandmarkHead", 1)?, head("LandmarkHead", 2)?],
        })
    }

    /// Raw network outputs: (loc [N,4], conf [N,2] softmaxed, landmarks [N,10]).
    /// `x` is BGR, 0..255, mean-subtracted, shape (1,3,H,W).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let feats = self.backbone.forward(x)?;
        let o1 = self.fpn_out[0].forward(&feats[0])?;
        let o2 = self.fpn_out[1].forward(&feats[1])?;
        let o3 = self.fpn_out[2].forward(&feats[2])?;
        let (_, _, h2, w2) = o2.dims4()?;
        let o2 = self.merge2.forward(&(o2 + resize_nearest(&o3, h2, w2)?)?)?;
        let (_, _, h1, w1) = o1.dims4()?;
        let o1 = self.merge1.forward(&(o1 + resize_nearest(&o2, h1, w1)?)?)?;
        let fs = [self.ssh[0].forward(&o1)?, self.ssh[1].forward(&o2)?, self.ssh[2].forward(&o3)?];
        let flat = |heads: &[Conv2d; 3], k: usize| -> Result<Tensor> {
            let parts = fs
                .iter()
                .zip(heads.iter())
                .map(|(f, h)| {
                    let y = h.forward(f)?.permute((0, 2, 3, 1))?.contiguous()?;
                    let n = y.elem_count() / k;
                    y.reshape((n, k))
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, 0)
        };
        let loc = flat(&self.bbox_head, 4)?;
        let conf = softmax_last(&flat(&self.class_head, 2)?)?;
        let landm = flat(&self.landmark_head, 10)?;
        Ok((loc, conf, landm))
    }

    /// Detect faces in an RGB image tensor (1,3,H,W) with values in 0..1.
    pub fn detect(&self, rgb: &Tensor, conf_threshold: f32, nms_threshold: f32) -> Result<Vec<Detection>> {
        let (_, _, h, w) = rgb.dims4()?;
        let dev = rgb.device();
        // RGB 0..1 -> BGR 0..255 minus mean (104, 117, 123)
        let bgr = Tensor::cat(&[rgb.narrow(1, 2, 1)?, rgb.narrow(1, 1, 1)?, rgb.narrow(1, 0, 1)?], 1)?;
        let mean = Tensor::new(&[104f32, 117., 123.], dev)?.to_dtype(rgb.dtype())?.reshape((1, 3, 1, 1))?;
        let x = (bgr * 255.0)?.broadcast_sub(&mean)?;
        let (loc, conf, landm) = self.forward(&x)?;
        let loc = loc.to_vec2::<f32>()?;
        let conf = conf.to_vec2::<f32>()?;
        let landm = landm.to_vec2::<f32>()?;
        let priors = prior_boxes(h, w);
        let mut dets = Vec::new();
        for (i, p) in priors.iter().enumerate() {
            let score = conf[i][1];
            if score <= conf_threshold {
                continue;
            }
            let l = &loc[i];
            let cx = p[0] + l[0] * 0.1 * p[2];
            let cy = p[1] + l[1] * 0.1 * p[3];
            let bw = p[2] * (l[2] * 0.2).exp();
            let bh = p[3] * (l[3] * 0.2).exp();
            let x0 = cx - bw / 2.0;
            let y0 = cy - bh / 2.0;
            let bbox = [x0 * w as f32, y0 * h as f32, (x0 + bw) * w as f32, (y0 + bh) * h as f32];
            let mut lm = [[0f32; 2]; 5];
            for (k, item) in lm.iter_mut().enumerate() {
                item[0] = (p[0] + landm[i][2 * k] * 0.1 * p[2]) * w as f32;
                item[1] = (p[1] + landm[i][2 * k + 1] * 0.1 * p[3]) * h as f32;
            }
            dets.push(Detection { bbox, score, landmarks: lm });
        }
        dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(nms(dets, nms_threshold))
    }
}

/// RetinaFace prior boxes (cx, cy, w, h), normalised.
pub fn prior_boxes(h: usize, w: usize) -> Vec<[f32; 4]> {
    let min_sizes: [[f32; 2]; 3] = [[16., 32.], [64., 128.], [256., 512.]];
    let steps = [8usize, 16, 32];
    let mut out = Vec::new();
    for (k, step) in steps.iter().enumerate() {
        let fh = h.div_ceil(*step);
        let fw = w.div_ceil(*step);
        for i in 0..fh {
            for j in 0..fw {
                for ms in min_sizes[k] {
                    out.push([(j as f32 + 0.5) * *step as f32 / w as f32, (i as f32 + 0.5) * *step as f32 / h as f32, ms / w as f32, ms / h as f32]);
                }
            }
        }
    }
    out
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

/// Greedy non-maximum suppression; `dets` must be sorted by descending score.
pub fn nms(dets: Vec<Detection>, thr: f32) -> Vec<Detection> {
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| iou(&k.bbox, &d.bbox) <= thr) {
            keep.push(d);
        }
    }
    keep
}
