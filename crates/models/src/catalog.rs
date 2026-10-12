//! Model catalog: versioned, hash-pinned manifests of downloadable models.

use serde::{Deserialize, Serialize};

use crate::convert::KeyRules;

pub const CATALOG_SCHEMA: u32 = 1;

/// The catalog compiled into this build (works offline).
pub const BUNDLED_CATALOG: &str = include_str!("../../../model-catalog/catalog.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    /// Role used by the adapter (`weights`, `wdn`, `restorer`, `detector`).
    pub role: String,
    pub filename: String,
    pub source_url: String,
    #[serde(default)]
    pub mirrors: Vec<String>,
    pub size_bytes: u64,
    pub sha256: String,
    /// `pth` (PyTorch checkpoint), `zip` (archive containing a checkpoint) or `safetensors`.
    pub format: String,
    /// For `zip`: the archive member to extract.
    #[serde(default)]
    pub member: Option<String>,
    #[serde(default)]
    pub convert: KeyRules,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Measurement {
    /// Hardware description, e.g. "4-core x86-64 cloud VM (AVX2), CPU backend".
    pub hardware: String,
    pub backend: String,
    /// Input megapixels processed per second.
    pub input_megapixels_per_second: f64,
    pub peak_memory_mb: f64,
    pub measured_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub display_name: String,
    pub publisher: String,
    pub summary: String,
    pub best_for: Vec<String>,
    pub weaknesses: Vec<String>,
    pub categories: Vec<String>,
    pub tasks: Vec<String>,
    pub architecture: String,
    /// Output scale factor (1 for same-size restoration).
    pub scale: u32,
    #[serde(default)]
    pub supports_denoise_strength: bool,
    pub input_types: Vec<String>,
    pub color_formats: Vec<String>,
    pub artifacts: Vec<Artifact>,
    pub download_size_bytes: u64,
    pub installed_size_bytes: u64,
    pub supported_platforms: Vec<String>,
    pub backends: Vec<String>,
    pub precisions: Vec<String>,
    pub minimum_ram_bytes: u64,
    /// Working memory per input megapixel processed at once (tiles), in MB.
    pub memory_per_megapixel_mb: f64,
    pub performance_tier: String,
    #[serde(default)]
    pub measurements: Vec<Measurement>,
    pub weights_license: String,
    pub code_license: String,
    pub commercial_use_allowed: bool,
    pub redistribution_allowed: bool,
    pub license_notes: String,
    pub attribution: Vec<String>,
    pub source_repository: String,
    pub training_data: String,
    pub provenance_reviewed_at: String,
    pub quality_status: String,
    #[serde(default)]
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Catalog {
    pub schema_version: u32,
    pub catalog_version: String,
    pub generated_at: String,
    pub models: Vec<ModelManifest>,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("catalog is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("catalog validation failed: {0}")]
    Invalid(String),
}

impl Catalog {
    pub fn parse(s: &str) -> Result<Self, CatalogError> {
        let c: Catalog = serde_json::from_str(s)?;
        c.validate()?;
        Ok(c)
    }

    pub fn bundled() -> Self {
        Self::parse(BUNDLED_CATALOG).expect("bundled catalog is validated by tests")
    }

    pub fn get(&self, id: &str) -> Option<&ModelManifest> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Structural validation: never accept placeholder hashes, insecure URLs or bogus sizes.
    pub fn validate(&self) -> Result<(), CatalogError> {
        let bad = |m: String| Err(CatalogError::Invalid(m));
        if self.schema_version != CATALOG_SCHEMA {
            return bad(format!("unsupported catalog schema {}", self.schema_version));
        }
        let mut ids = std::collections::HashSet::new();
        for m in &self.models {
            if m.schema_version != CATALOG_SCHEMA {
                return bad(format!("{}: unsupported manifest schema", m.id));
            }
            if !ids.insert(&m.id) {
                return bad(format!("duplicate model id {}", m.id));
            }
            if m.id.is_empty() || m.id.len() > 80 || !m.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-' || c == '_') {
                return bad(format!("invalid model id `{}`", m.id));
            }
            if m.artifacts.is_empty() {
                return bad(format!("{}: no artifacts", m.id));
            }
            let mut total = 0u64;
            let mut roles = std::collections::HashSet::new();
            for a in &m.artifacts {
                if !roles.insert(&a.role) {
                    return bad(format!("{}: duplicate artifact role {}", m.id, a.role));
                }
                if a.sha256.len() != 64 || !a.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
                    return bad(format!("{}: artifact {} has an invalid sha256", m.id, a.filename));
                }
                for u in std::iter::once(&a.source_url).chain(a.mirrors.iter()) {
                    if !u.starts_with("https://") {
                        return bad(format!("{}: artifact URLs must use https ({u})", m.id));
                    }
                }
                if a.size_bytes == 0 || a.size_bytes > 8 << 30 {
                    return bad(format!("{}: implausible artifact size", m.id));
                }
                if !matches!(a.format.as_str(), "pth" | "zip" | "safetensors") {
                    return bad(format!("{}: unsupported artifact format {}", m.id, a.format));
                }
                if a.format == "zip" && a.member.is_none() {
                    return bad(format!("{}: zip artifact needs a member", m.id));
                }
                if a.filename.contains('/') || a.filename.contains('\\') || a.filename.contains("..") {
                    return bad(format!("{}: artifact file name must not contain paths", m.id));
                }
                total += a.size_bytes;
            }
            if total != m.download_size_bytes {
                return bad(format!("{}: download_size_bytes does not equal the sum of artifacts", m.id));
            }
            if m.weights_license.trim().is_empty() || m.weights_license.contains("REPLACE") {
                return bad(format!("{}: weights license missing", m.id));
            }
            if m.scale == 0 || m.scale > 8 {
                return bad(format!("{}: invalid scale", m.id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_is_valid() {
        let c = Catalog::bundled();
        assert!(c.models.len() >= 10);
        for m in &c.models {
            assert!(m.redistribution_allowed || !m.recommended, "{}", m.id);
        }
        // Every architecture referenced is one the app ships an adapter for.
        for m in &c.models {
            assert!(
                ["srvgg_compact", "rrdbnet", "ffdnet", "drunet", "nafnet", "rife", "gfpgan"].contains(&m.architecture.as_str()),
                "{}",
                m.architecture
            );
        }
    }

    #[test]
    fn rejects_placeholder_hash() {
        let mut c = Catalog::bundled();
        c.models[0].artifacts[0].sha256 = "REPLACE_WITH_REAL_HASH".into();
        assert!(c.validate().is_err());
        let mut c = Catalog::bundled();
        c.models[0].artifacts[0].source_url = "http://example.org/x".into();
        assert!(c.validate().is_err());
        let mut c = Catalog::bundled();
        c.models[0].artifacts[0].filename = "../evil".into();
        assert!(c.validate().is_err());
    }
}
