//! Unified model API used by the processing pipeline.

use candle_core::{DType, Device, Result};
use serde::{Deserialize, Serialize};

use crate::arch::{drunet::DruNet, ffdnet::FfdNet, gfpgan::Gfpgan, nafnet::NafNet, retinaface::RetinaFace, rife::Rife, rrdb::RrdbNet, srvgg::SrVggCompact};
use crate::image::RgbImage;
use crate::tiling::{pad_to_multiple, process_tiled};
use crate::weights::Weights;

/// Network architectures ClipFX Enhance ships reviewed adapters for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    SrVggCompact,
    RrdbNet,
    FfdNet,
    DruNet,
    NafNet,
    Rife,
    RetinaFace,
    Gfpgan,
}

impl Architecture {
    pub fn id(self) -> &'static str {
        match self {
            Architecture::SrVggCompact => "srvgg_compact",
            Architecture::RrdbNet => "rrdbnet",
            Architecture::FfdNet => "ffdnet",
            Architecture::DruNet => "drunet",
            Architecture::NafNet => "nafnet",
            Architecture::Rife => "rife",
            Architecture::RetinaFace => "retinaface",
            Architecture::Gfpgan => "gfpgan",
        }
    }

    /// Identify the architecture from tensor names (used to validate converted
    /// checkpoints and user-imported ESRGAN-family models).
    pub fn detect(w: &Weights) -> Option<Self> {
        if w.contains("conv_first.weight") && w.contains("conv_up1.weight") {
            Some(Architecture::RrdbNet)
        } else if w.contains("body.0.weight") && w.contains("body.1.weight") && !w.contains("conv_first.weight") {
            Some(Architecture::SrVggCompact)
        } else if w.contains("model.0.weight") && !w.contains("m_head.weight") {
            Some(Architecture::FfdNet)
        } else if w.contains("m_head.weight") && w.contains("m_down1.4.weight") {
            Some(Architecture::DruNet)
        } else if w.contains("intro.weight") && w.contains("ending.weight") {
            Some(Architecture::NafNet)
        } else if w.contains("block0.conv0.0.0.weight") && w.contains("encode.cnn0.weight") {
            Some(Architecture::Rife)
        } else if w.contains("ClassHead.0.conv1x1.weight") {
            Some(Architecture::RetinaFace)
        } else if w.contains("stylegan_decoder.constant_input.weight") {
            Some(Architecture::Gfpgan)
        } else {
            None
        }
    }

    /// Default tile size (input pixels) and context padding.
    pub fn default_tiling(self) -> (usize, usize) {
        match self {
            Architecture::SrVggCompact => (512, 16),
            Architecture::RrdbNet => (384, 24),
            Architecture::FfdNet => (1024, 16),
            Architecture::DruNet => (512, 32),
            Architecture::NafNet => (384, 32),
            _ => (0, 0),
        }
    }
}

/// Numeric precision for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Fp32,
    Fp16,
}

impl Precision {
    pub fn dtype(self, device: &Device) -> DType {
        match (self, device) {
            // Half precision is only beneficial (and well supported) on GPU backends.
            (Precision::Fp16, Device::Cpu) => DType::F32,
            (Precision::Fp16, _) => DType::F16,
            _ => DType::F32,
        }
    }
}

enum ImageNet {
    SrVgg(SrVggCompact),
    Rrdb(RrdbNet),
    Ffd(FfdNet),
    Dru(DruNet),
    Naf(NafNet),
}

/// Parameters for a single image-model invocation.
#[derive(Debug, Clone, Copy)]
pub struct ImageParams {
    /// Noise level on a 0..1 scale for noise-conditioned denoisers (FFDNet / DRUNet).
    pub noise_sigma: f32,
}

impl Default for ImageParams {
    fn default() -> Self {
        Self { noise_sigma: 15.0 / 255.0 }
    }
}

/// An image-to-image model (super-resolution, denoise, deblur).
pub struct ImageModel {
    net: ImageNet,
    pub arch: Architecture,
    device: Device,
    dtype: DType,
    pub tile: usize,
    pub tile_pad: usize,
    #[cfg(feature = "gpu")]
    gpu: Option<(std::sync::Arc<crate::gpu::Gpu>, crate::gpu::nets::GpuNet)>,
}

fn convert_weights(w: &Weights, dtype: DType) -> Result<Weights> {
    w.map_dtype(dtype)
}

impl ImageModel {
    pub fn load(w: &Weights, precision: Precision) -> Result<Self> {
        let arch = Architecture::detect(w).ok_or_else(|| candle_core::Error::Msg("unrecognised model architecture".into()))?;
        let device = w.device().clone();
        let dtype = precision.dtype(&device);
        let w = convert_weights(w, dtype)?;
        let net = match arch {
            Architecture::SrVggCompact => ImageNet::SrVgg(SrVggCompact::load(&w)?),
            Architecture::RrdbNet => ImageNet::Rrdb(RrdbNet::load(&w)?),
            Architecture::FfdNet => ImageNet::Ffd(FfdNet::load(&w)?),
            Architecture::DruNet => ImageNet::Dru(DruNet::load(&w)?),
            Architecture::NafNet => ImageNet::Naf(NafNet::load(&w)?),
            other => candle_core::bail!("{} is not an image-to-image model", other.id()),
        };
        let (tile, tile_pad) = arch.default_tiling();
        Ok(Self {
            net,
            arch,
            device,
            dtype,
            tile,
            tile_pad,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    /// Run this model on a GPU (any vendor) instead of the CPU, with tiles as large as the
    /// GPU accepts.
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self, gpu: std::sync::Arc<crate::gpu::Gpu>) -> Result<()> {
        self.use_gpu_with_budget(gpu, 0)
    }

    /// Like [`use_gpu`](Self::use_gpu), keeping one tile's working memory within
    /// `vram_budget_mb` of video memory (0 = no limit beyond what the GPU allows).
    #[cfg(feature = "gpu")]
    pub fn use_gpu_with_budget(&mut self, gpu: std::sync::Arc<crate::gpu::Gpu>, vram_budget_mb: u64) -> Result<()> {
        use crate::gpu::nets::*;
        let err = |e: String| candle_core::Error::Msg(format!("GPU: {e}"));
        let net = match &self.net {
            ImageNet::SrVgg(n) => GpuNet::SrVgg(GSrVgg::new(&gpu, n).map_err(err)?),
            ImageNet::Rrdb(n) => GpuNet::Rrdb(Box::new(GRrdb::new(&gpu, n).map_err(err)?)),
            ImageNet::Ffd(n) => GpuNet::Ffd(GFfd::new(&gpu, n).map_err(err)?),
            ImageNet::Dru(n) => GpuNet::Dru(Box::new(GDru::new(&gpu, n).map_err(err)?)),
            ImageNet::Naf(n) => GpuNet::Naf(Box::new(GNaf::new(&gpu, n).map_err(err)?)),
        };
        // Big tiles keep the GPU busy and cut per-tile overhead, so start from 2048 px (a
        // whole 1080p frame in one pass for most models) and shrink until the largest tensor
        // fits in one GPU buffer and, with a budget, the working set (about 4x the largest
        // tensor) fits in the allowed video memory.
        let limit = gpu.max_binding;
        let budget = if vram_budget_mb > 0 { vram_budget_mb << 20 } else { u64::MAX };
        let mut t = 2048usize;
        while t > 64 && {
            let big = net.largest_tensor_bytes(t + 2 * self.tile_pad, t + 2 * self.tile_pad);
            big > limit || big.saturating_mul(4) > budget
        } {
            t = t * 7 / 8;
        }
        self.tile = t;
        self.gpu = Some((gpu, net));
        Ok(())
    }

    /// Name of the GPU this model runs on, if any.
    pub fn gpu_name(&self) -> Option<String> {
        #[cfg(feature = "gpu")]
        if let Some((g, _)) = &self.gpu {
            return Some(g.info.name.clone());
        }
        None
    }

    /// Output scale factor relative to input.
    pub fn scale(&self) -> usize {
        match &self.net {
            ImageNet::SrVgg(n) => n.scale,
            ImageNet::Rrdb(n) => n.scale,
            _ => 1,
        }
    }

    fn pad_multiple(&self) -> usize {
        match &self.net {
            ImageNet::Rrdb(n) => n.pad_multiple(),
            _ => 1,
        }
    }

    /// Run the network on a single crop (no tiling).
    pub fn run(&self, img: &RgbImage, p: &ImageParams) -> Result<RgbImage> {
        let (y, padded) = self.run_raw(img, p)?;
        self.finish(y, img, &padded)
    }

    /// The network output before clamping and cropping (used by parity tests).
    #[doc(hidden)]
    pub fn run_raw(&self, img: &RgbImage, p: &ImageParams) -> Result<(candle_core::Tensor, RgbImage)> {
        let m = self.pad_multiple();
        let padded = pad_to_multiple(img, m);
        let x = padded.to_tensor(&self.device, self.dtype)?;
        #[cfg(feature = "gpu")]
        if let Some((gpu, net)) = &self.gpu {
            let x = if self.dtype == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
            let y = net.forward(gpu, &x, p.noise_sigma).map_err(|e| candle_core::Error::Msg(format!("GPU: {e}")))?;
            return Ok((y, padded));
        }
        let y = match &self.net {
            ImageNet::SrVgg(n) => n.forward(&x)?,
            ImageNet::Rrdb(n) => n.forward(&x)?,
            ImageNet::Ffd(n) => n.forward(&x, p.noise_sigma)?,
            ImageNet::Dru(n) => n.forward(&x, p.noise_sigma)?,
            ImageNet::Naf(n) => n.forward(&x)?,
        };
        Ok((y, padded))
    }

    fn finish(&self, y: candle_core::Tensor, img: &RgbImage, padded: &RgbImage) -> Result<RgbImage> {
        let y = y.clamp(0f32, 1f32)?;
        let s = self.scale();
        let y = if padded.width != img.width || padded.height != img.height { y.narrow(2, 0, img.height * s)?.narrow(3, 0, img.width * s)? } else { y };
        RgbImage::from_tensor(&y)
    }

    /// Run with tiling.
    pub fn process(&self, img: &RgbImage, p: &ImageParams) -> Result<RgbImage> {
        process_tiled(img, self.scale(), self.tile, self.tile_pad, |crop| self.run(crop, p))
    }
}

/// Frame interpolation model.
pub struct InterpModel {
    net: Rife,
    device: Device,
    dtype: DType,
    #[cfg(feature = "gpu")]
    gpu: Option<(std::sync::Arc<crate::gpu::Gpu>, crate::gpu::nets::GRife)>,
}

impl InterpModel {
    pub fn load(w: &Weights, precision: Precision) -> Result<Self> {
        let device = w.device().clone();
        let dtype = precision.dtype(&device);
        let w = convert_weights(w, dtype)?;
        Ok(Self {
            net: Rife::load(&w)?,
            device,
            dtype,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    /// Run RIFE on a cross-vendor GPU (Vulkan / DirectX 12 / Metal) instead of the CPU.
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self, gpu: std::sync::Arc<crate::gpu::Gpu>) -> Result<()> {
        let net = crate::gpu::nets::GRife::new(&gpu, &self.net).map_err(candle_core::Error::msg)?;
        self.gpu = Some((gpu, net));
        Ok(())
    }

    /// Whether frames are synthesized on the GPU.
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.gpu.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }

    #[cfg(feature = "gpu")]
    fn interpolate_gpu(&self, a: &RgbImage, b: &RgbImage, t: f32, scale: f64) -> Option<RgbImage> {
        use crate::gpu::nets::Planar;
        let (gpu, net) = self.gpu.as_ref()?;
        let run = || -> std::result::Result<RgbImage, String> {
            let (h, w) = (a.height, a.width);
            let m = ((64.0 / scale).max(64.0)) as usize;
            let (ph, pw) = ((m - h % m) % m, (m - w % m) % m);
            let pa = Planar::from_tensor(&a.to_tensor(&Device::Cpu, DType::F32).map_err(|e| e.to_string())?)?.pad_replicate(pw, ph);
            let pb = Planar::from_tensor(&b.to_tensor(&Device::Cpu, DType::F32).map_err(|e| e.to_string())?)?.pad_replicate(pw, ph);
            let y = net.forward(gpu, &pa, &pb, t, scale)?.to_tensor()?;
            let y = y.narrow(2, 0, h).and_then(|y| y.narrow(3, 0, w)).map_err(|e| e.to_string())?;
            RgbImage::from_tensor(&y).map_err(|e| e.to_string())
        };
        match run() {
            Ok(img) => Some(img),
            Err(e) => {
                log::warn!("RIFE on the GPU failed ({e}); using the CPU for this frame");
                None
            }
        }
    }

    /// Synthesize a frame at `t` between `a` and `b`.
    pub fn interpolate(&self, a: &RgbImage, b: &RgbImage, t: f32, scale: f64) -> Result<RgbImage> {
        #[cfg(feature = "gpu")]
        if let Some(img) = self.interpolate_gpu(a, b, t, scale) {
            return Ok(img);
        }
        let x0 = a.to_tensor(&self.device, self.dtype)?;
        let x1 = b.to_tensor(&self.device, self.dtype)?;
        let y = self.net.interpolate(&x0, &x1, t, scale)?;
        RgbImage::from_tensor(&y)
    }

    /// Synthesize several timesteps, re-using the uploaded input tensors.
    pub fn interpolate_many(&self, a: &RgbImage, b: &RgbImage, ts: &[f32], scale: f64) -> Result<Vec<RgbImage>> {
        if self.on_gpu() {
            return ts.iter().map(|t| self.interpolate(a, b, *t, scale)).collect();
        }
        let x0 = a.to_tensor(&self.device, self.dtype)?;
        let x1 = b.to_tensor(&self.device, self.dtype)?;
        ts.iter().map(|t| RgbImage::from_tensor(&self.net.interpolate(&x0, &x1, *t, scale)?)).collect()
    }
}

/// Face detection + restoration models.
pub struct FaceModels {
    pub detector: RetinaFace,
    pub restorer: Gfpgan,
    pub device: Device,
    #[cfg(feature = "gpu")]
    gpu_restorer: Option<(std::sync::Arc<crate::gpu::Gpu>, crate::gpu::nets::GGfpgan)>,
    #[cfg(feature = "gpu")]
    gpu_detector: Option<crate::gpu::nets::GRetinaFace>,
}

impl FaceModels {
    pub fn load(detector: &Weights, restorer: &Weights) -> Result<Self> {
        Ok(Self {
            detector: RetinaFace::load(detector)?,
            restorer: Gfpgan::load(restorer)?,
            device: restorer.device().clone(),
            #[cfg(feature = "gpu")]
            gpu_restorer: None,
            #[cfg(feature = "gpu")]
            gpu_detector: None,
        })
    }

    /// Restore faces on a cross-vendor GPU (Vulkan / DirectX 12 / Metal) instead of the CPU.
    #[cfg(feature = "gpu")]
    pub fn use_gpu(&mut self, gpu: std::sync::Arc<crate::gpu::Gpu>) -> Result<()> {
        let det = crate::gpu::nets::GRetinaFace::new(&gpu, &self.detector).map_err(candle_core::Error::msg)?;
        let net = crate::gpu::nets::GGfpgan::new(&gpu, &self.restorer).map_err(candle_core::Error::msg)?;
        self.gpu_detector = Some(det);
        self.gpu_restorer = Some((gpu, net));
        Ok(())
    }

    /// Detect faces in an RGB image tensor (1,3,H,W) in 0..1. Uses the GPU when attached, else (or on a GPU failure)
    /// the CPU; pre- and post-processing are shared.
    pub fn detect(&self, rgb: &candle_core::Tensor, conf_threshold: f32, nms_threshold: f32) -> Result<Vec<crate::arch::retinaface::Detection>> {
        #[cfg(feature = "gpu")]
        if let (Some(det), Some((gpu, _))) = (&self.gpu_detector, &self.gpu_restorer) {
            use crate::arch::retinaface as rf;
            let run = || -> Result<Vec<rf::Detection>> {
                let (_, _, h, w) = rgb.dims4()?;
                let x = rf::preprocess(&rgb.to_dtype(DType::F32)?)?;
                let p = crate::gpu::nets::Planar::from_tensor(&x).map_err(candle_core::Error::msg)?;
                let o = det.forward(gpu, &p).map_err(candle_core::Error::msg)?;
                let (loc, conf, landm) = rf::heads_to_outputs(&o[0..3], &o[3..6], &o[6..9])?;
                rf::decode(&loc, &conf, &landm, h, w, conf_threshold, nms_threshold)
            };
            match run() {
                Ok(d) => return Ok(d),
                Err(e) => log::warn!("face detection on the GPU failed ({e}); using the CPU for this frame"),
            }
        }
        self.detector.detect(rgb, conf_threshold, nms_threshold)
    }

    /// Whether faces are restored on the GPU.
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.gpu_restorer.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Restore an aligned 512×512 face (1,3,512,512) in 0..1. Uses the GPU when attached, else (or on a GPU
    /// failure) the CPU.
    pub fn restore(&self, face: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        #[cfg(feature = "gpu")]
        if let Some((gpu, net)) = &self.gpu_restorer {
            let run = || -> std::result::Result<candle_core::Tensor, String> {
                let p = crate::gpu::nets::Planar::from_tensor(&face.to_dtype(DType::F32).map_err(|e| e.to_string())?)?;
                net.forward(gpu, &p)?.to_tensor()
            };
            match run() {
                Ok(t) => return Ok(t),
                Err(e) => log::warn!("face restoration on the GPU failed ({e}); using the CPU for this face"),
            }
        }
        self.restorer.forward(face)
    }
}
