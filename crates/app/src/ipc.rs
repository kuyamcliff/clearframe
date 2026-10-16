//! Values the desktop UI asks for at startup, shared with the UI test fixtures.

use clearframe_pipeline::spec::*;

/// Default settings of every operation, the default export format and the grain looks.
pub fn operation_defaults() -> serde_json::Value {
    let grain: Vec<GrainSettings> =
        [GrainPreset::Fine, GrainPreset::Film, GrainPreset::Coarse, GrainPreset::Monochrome, GrainPreset::Color].iter().map(|p| GrainSettings::preset(*p)).collect();
    serde_json::json!({
        "deinterlace": Operation::Deinterlace(Default::default()),
        "denoise": Operation::Denoise(Default::default()),
        "deblur": Operation::Deblur(Default::default()),
        "enhance": Operation::Enhance(Default::default()),
        "face_restore": Operation::FaceRestore(Default::default()),
        "interpolate": Operation::Interpolate(Default::default()),
        "stabilize": Operation::Stabilize(Default::default()),
        "color": Operation::Color(Default::default()),
        "sharpen": Operation::Sharpen(Default::default()),
        "grain": Operation::Grain(Default::default()),
        "resize": Operation::Resize(Default::default()),
        "output": OutputSettings::default(),
        "grain_presets": grain,
    })
}
