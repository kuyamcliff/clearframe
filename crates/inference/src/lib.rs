//! ClipFX Enhance inference: reviewed, pure-Rust model adapters on top of candle.
//!
//! Model files are never executed; weights are loaded from safetensors files
//! produced by the installer's restricted checkpoint converter.
pub mod arch;
pub mod device;
pub mod face;
pub mod fastconv;
pub mod image;
pub mod model;
pub mod ops;
pub mod tiling;
pub mod weights;

pub use candle_core::{DType, Device};
pub use image::RgbImage;
pub use model::{Architecture, ImageModel, ImageParams, InterpModel, Precision};
pub use weights::Weights;
