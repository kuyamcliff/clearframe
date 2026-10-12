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
        Ok(Self { net, arch, device, dtype, tile, tile_pad })
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
        let m = self.pad_multiple();
        let padded = pad_to_multiple(img, m);
        let x = padded.to_tensor(&self.device, self.dtype)?;
        let y = match &self.net {
            ImageNet::SrVgg(n) => n.forward(&x)?,
            ImageNet::Rrdb(n) => n.forward(&x)?,
            ImageNet::Ffd(n) => n.forward(&x, p.noise_sigma)?,
            ImageNet::Dru(n) => n.forward(&x, p.noise_sigma)?,
            ImageNet::Naf(n) => n.forward(&x)?,
        };
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
}

impl InterpModel {
    pub fn load(w: &Weights, precision: Precision) -> Result<Self> {
        let device = w.device().clone();
        let dtype = precision.dtype(&device);
        let w = convert_weights(w, dtype)?;
        Ok(Self { net: Rife::load(&w)?, device, dtype })
    }

    /// Synthesize a frame at `t` between `a` and `b`.
    pub fn interpolate(&self, a: &RgbImage, b: &RgbImage, t: f32, scale: f64) -> Result<RgbImage> {
        let x0 = a.to_tensor(&self.device, self.dtype)?;
        let x1 = b.to_tensor(&self.device, self.dtype)?;
        let y = self.net.interpolate(&x0, &x1, t, scale)?;
        RgbImage::from_tensor(&y)
    }

    /// Synthesize several timesteps, re-using the uploaded input tensors.
    pub fn interpolate_many(&self, a: &RgbImage, b: &RgbImage, ts: &[f32], scale: f64) -> Result<Vec<RgbImage>> {
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
}

impl FaceModels {
    pub fn load(detector: &Weights, restorer: &Weights) -> Result<Self> {
        Ok(Self { detector: RetinaFace::load(detector)?, restorer: Gfpgan::load(restorer)?, device: restorer.device().clone() })
    }
}
