//! Colour matrix, range and metadata handling.

use serde::{Deserialize, Serialize};

use crate::probe::VideoStream;

/// YUV <-> RGB conversion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YuvMatrix {
    Bt601,
    Bt709,
    Bt2020,
    Fcc,
    Smpte240m,
}

impl YuvMatrix {
    /// Name accepted by the `scale` filter's `in_color_matrix`/`out_color_matrix`.
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            YuvMatrix::Bt601 => "bt601",
            YuvMatrix::Bt709 => "bt709",
            YuvMatrix::Bt2020 => "bt2020",
            YuvMatrix::Fcc => "fcc",
            YuvMatrix::Smpte240m => "smpte240m",
        }
    }
    /// Value for the `-colorspace` output option.
    pub fn colorspace_tag(self) -> &'static str {
        match self {
            YuvMatrix::Bt601 => "smpte170m",
            YuvMatrix::Bt709 => "bt709",
            YuvMatrix::Bt2020 => "bt2020nc",
            YuvMatrix::Fcc => "fcc",
            YuvMatrix::Smpte240m => "smpte240m",
        }
    }
    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "bt709" => YuvMatrix::Bt709,
            "smpte170m" | "bt470bg" | "bt601" => YuvMatrix::Bt601,
            "bt2020nc" | "bt2020c" | "bt2020" => YuvMatrix::Bt2020,
            "fcc" => YuvMatrix::Fcc,
            "smpte240m" => YuvMatrix::Smpte240m,
            _ => return None,
        })
    }
    /// Conventional default when a stream is untagged.
    pub fn guess(width: u32, height: u32) -> Self {
        if width >= 1280 || height > 576 {
            YuvMatrix::Bt709
        } else {
            YuvMatrix::Bt601
        }
    }
}

/// Output colour description written into the encoded stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorTags {
    pub matrix: YuvMatrix,
    pub full_range: bool,
    pub primaries: Option<String>,
    pub transfer: Option<String>,
    pub mastering_display: Option<String>,
    pub content_light: Option<String>,
}

impl ColorTags {
    pub fn sdr_bt709() -> Self {
        Self { matrix: YuvMatrix::Bt709, full_range: false, primaries: Some("bt709".into()), transfer: Some("bt709".into()), mastering_display: None, content_light: None }
    }

    /// Derive tags from a probed stream (keeping what is valid, guessing what is missing).
    pub fn from_stream(v: &VideoStream) -> Self {
        let matrix = v.color.space.as_deref().and_then(YuvMatrix::from_tag).unwrap_or_else(|| YuvMatrix::guess(v.width, v.height));
        let is_rgb = v.pix_fmt.starts_with("gbr") || v.pix_fmt.starts_with("rgb") || v.pix_fmt.starts_with("bgr");
        let full_range = match v.color.range.as_deref() {
            Some("pc") | Some("jpeg") => true,
            Some("tv") | Some("mpeg") => false,
            _ => is_rgb || v.pix_fmt.starts_with("yuvj"),
        };
        Self {
            matrix,
            full_range,
            primaries: v.color.primaries.clone(),
            transfer: v.color.transfer.clone(),
            mastering_display: v.color.mastering_display.clone(),
            content_light: v.color.content_light.clone(),
        }
    }

    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_deref(), Some("smpte2084") | Some("arib-std-b67"))
    }

    /// `-colorspace/-color_primaries/-color_trc/-color_range` output options.
    pub fn ffmpeg_args(&self) -> Vec<String> {
        let mut a = vec!["-colorspace".to_string(), self.matrix.colorspace_tag().to_string()];
        if let Some(p) = &self.primaries {
            a.extend(["-color_primaries".to_string(), p.clone()]);
        }
        if let Some(t) = &self.transfer {
            a.extend(["-color_trc".to_string(), t.clone()]);
        }
        a.extend(["-color_range".to_string(), if self.full_range { "pc".into() } else { "tv".into() }]);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_mapping() {
        assert_eq!(YuvMatrix::from_tag("bt470bg"), Some(YuvMatrix::Bt601));
        assert_eq!(YuvMatrix::guess(720, 480), YuvMatrix::Bt601);
        assert_eq!(YuvMatrix::guess(1920, 1080), YuvMatrix::Bt709);
        assert_eq!(YuvMatrix::Bt2020.colorspace_tag(), "bt2020nc");
    }

    #[test]
    fn args() {
        let a = ColorTags::sdr_bt709().ffmpeg_args();
        assert_eq!(a, vec!["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]);
    }
}
